"""
LLM Client Interface for local Ollama usage.

Used for fast summarization (PageIndex) and structured extraction (FactTable)
using the user-specified Qwen model.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


def get_chat_model(temperature: float = 0.0, is_vlm: bool = False):
    """
    Dynamically loads the Chat Model based on ENV variables.
    Supports LLM_PROVIDER / LLM_MODEL or VLM_PROVIDER / VLM_MODEL.
    """
    prefix = "VLM" if is_vlm else "LLM"
    provider = os.environ.get(f"{prefix}_PROVIDER", "ollama").lower()
    model = os.environ.get(f"{prefix}_MODEL", "qwen2.5:latest" if not is_vlm else "llava")
    
    logger.info(f"Initializing {prefix} with provider='{provider}', model='{model}'")
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    else:  # Default to Ollama
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            # Fallback for older environments before the dependency is updated
            from langchain_community.chat_models import ChatOllama
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)


class LLMClient:
    """Dynamic LLM client wrapping LangChain for summarization and JSON extraction."""

    def __init__(self):
        self.llm = get_chat_model(temperature=0.0, is_vlm=False)

    def generate(self, prompt: str, system_prompt: str = "", format_type: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        kwargs = {}
        provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        if format_type == "json":
            if provider == "ollama":
                kwargs["format"] = "json"
            elif provider == "openai":
                kwargs["response_format"] = {"type": "json_object"}
                
        try:
            response = self.llm.invoke(messages, **kwargs)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    def generate_summary(self, text: str) -> str:
        if not text.strip():
            return "No text available to summarize."
        sys_prompt = "You are a concise summarizer. Output exactly 2-3 short sentences capturing the core concepts."
        prompt = f"Summarize the following section text:\n\n{text}"
        return self.generate(prompt=prompt, system_prompt=sys_prompt)

    def extract_facts(self, text: str, doc_mode: str = "financial") -> list[dict[str, str]]:
        sys_prompt = """You are a strict data extraction bot.
Extract numerical facts and metrics from the text.
Output MUST be a JSON object containing a "facts" array, where each object has these exact keys:
"metric_name" (string, e.g. "Total Revenue"),
"metric_value" (string, e.g. "$4.2B", include units),
"date_context" (string, e.g. "Q3 2024", or "" if none).
If no facts are found, return {"facts": []}. Do not include markdown formatting or explanations, ONLY output raw JSON."""

        prompt = f"Text to extract from:\n\n{text}"
        response_text = self.generate(prompt=prompt, system_prompt=sys_prompt, format_type="json")
        if not response_text:
            return []
            
        try:
            # Clean possible markdown block if LLM ignored the format instructions
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
                
            data = json.loads(response_text)
            if isinstance(data, dict) and "facts" in data:
                return [f for f in data["facts"] if isinstance(f, dict) and "metric_name" in f and "metric_value" in f]
            elif isinstance(data, list):
                return [f for f in data if isinstance(f, dict) and "metric_name" in f and "metric_value" in f]
            return []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM JSON response. Raw output: {response_text}")
            return []
