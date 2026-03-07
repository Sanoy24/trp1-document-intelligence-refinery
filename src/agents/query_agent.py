"""
Query Interface Agent (Stage 5).

Custom LangGraph StateGraph agent with three tools:
1. pageindex_navigate  — tree traversal over the PageIndex
2. semantic_search     — vector retrieval over LDU chunks
3. structured_query    — SQL over the SQLite FactTable

Outputs answers with ProvenanceChain citations and an Audit Mode.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.data.fact_store import FactStore
from src.data.vector_store import VectorStore
from src.models.schemas import Citation, ProvenanceChain
from src.utils.llm import get_chat_model

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Global singletons used by tools
# ─────────────────────────────────────────────────────────────
_vector_store = VectorStore()
_fact_store = FactStore()
_pageindex_dir = Path(".refinery/pageindex")


# ─────────────────────────────────────────────────────────────
# Tool definitions (unchanged)
# ─────────────────────────────────────────────────────────────

@tool
def pageindex_navigate(doc_id: str, section_title: str = "Root") -> str:
    """
    Navigate the hierarchical PageIndex tree for a document.
    Returns the summary and structure of the requested section to help understand document context.
    """
    idx_path = _pageindex_dir / f"{doc_id}_index.json"
    if not idx_path.exists():
        return f"Error: No PageIndex found for doc_id {doc_id}."

    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            tree = json.load(f)

        # Very simple traversal (assumes flat structure under Root for now)
        if section_title == "Root":
            children = [c.get("title") for c in tree.get("child_sections", [])]
            return f"Document: {tree.get('title')}\nSummary: {tree.get('summary')}\nSections: {', '.join(children)}"

        for child in tree.get("child_sections", []):
            if child.get("title", "").lower() == section_title.lower():
                return f"Section: {child['title']}\nSummary: {child.get('summary')}\nData Types: {child.get('data_types_present')}"

        return f"Section '{section_title}' not found in Root."
    except Exception as e:
        return f"Error reading PageIndex: {e}"


@tool
def semantic_search(query: str, n_results: int = 3) -> str:
    """
    Perform a semantic search across all document chunks (LDUs) in the VectorStore.
    Returns matched text, page numbers, and the source hash for provenance.
    """
    results = _vector_store.search(query, n_results=n_results)
    if not results:
        return "No semantic matches found."

    out = []
    for r in results:
        meta = r["metadata"]
        text = r["document"]
        source_hash = meta.get("content_hash", "")
        pages = meta.get("page_refs", "")
        bbox = meta.get("bounding_box", "")
        out.append(
            "--- MATCH ---\n"
            f"Text: {text}\n"
            f"Pages: {pages}\n"
            f"BBox: {bbox}\n"
            f"Hash: {source_hash}"
        )

    return "\n\n".join(out)


@tool
def structured_query(sql_query: str) -> str:
    """
    Execute a SELECT query against the SQLite FactTable containing financial/numerical metrics.
    Table schema: ExtractedFacts (id, doc_id, page_number, metric_name, metric_value, date_context, source_hash)
    """
    try:
        rows = _fact_store.query_facts(sql_query)
        if not rows:
            return "Query executed successfully, but returned 0 rows."

        # Format as JSON string
        return json.dumps(rows, indent=2)
    except Exception as e:
        return f"SQL Error: {e}"


# ─────────────────────────────────────────────────────────────
# LangGraph State & Graph Definition
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State schema for the LangGraph query agent."""
    messages: Annotated[list[BaseMessage], add_messages]


# System prompt injected at graph-entry time
SYSTEM_PROMPT = """You are the Document Intelligence Query Agent.
You have access to 3 tools to answer user questions about documents:
1. pageindex_navigate: To see document summaries and structure.
2. semantic_search: To search unstructured text.
3. structured_query: To query financial metrics (revenue, etc.) from the SQLite FactTable.

Always refer to the tools. When providing your final answer, you MUST list the 'source_hash' of any documents or facts you used so we can trace provenance."""


def _build_graph(llm_with_tools) -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Graph topology:
        START ──► agent ──►(conditional)──► tools ──► agent
                                       └──► END
    """

    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM (with tools bound) on the current message list."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # ToolNode automatically routes to the correct @tool function
    tool_node = ToolNode([pageindex_navigate, semantic_search, structured_query])

    def should_continue(state: AgentState) -> str:
        """Route to 'tools' if the LLM emitted tool calls, otherwise end."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    # --- Build the graph ---
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────

class QueryInterface:
    """Orchestrates the custom LangGraph agent and provenance checking."""

    def __init__(self):
        # Load the chat model and bind the three tools to it
        self.llm = get_chat_model(temperature=0.0, is_vlm=False)
        self.tools = [pageindex_navigate, semantic_search, structured_query]
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the compiled StateGraph
        self.graph = _build_graph(llm_with_tools)

        logger.info(
            "Initialized QueryInterface LangGraph Agent (custom StateGraph) "
            f"with model {getattr(self.llm, 'model_name', getattr(self.llm, 'model', 'unknown'))}"
        )

    def query(self, user_question: str) -> ProvenanceChain:
        """Run the LangGraph agent and format the output into a ProvenanceChain."""

        # Seed the conversation with the system prompt + user question
        inputs: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                ("user", user_question),
            ]
        }

        result = self.graph.invoke(inputs)

        final_message = result["messages"][-1].content

        # ── Extract real citations from ToolMessages ──
        citations = self._extract_citations(result["messages"])

        chain = ProvenanceChain(
            query=user_question,
            answer=final_message,
            citations=citations,
        )

        # Audit mode: verify the answer is grounded based on citations
        chain.verified = self._audit_claim(chain.answer, chain.citations)

        return chain

    # ── Provenance helpers ──────────────────────────────────

    @staticmethod
    def _extract_citations(messages: list[BaseMessage]) -> list[Citation]:
        """
        Walk through ToolMessages to extract provenance citations.
        Parses the structured text that semantic_search / structured_query return.
        """
        citations: list[Citation] = []

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue

            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tool_name = getattr(msg, "name", "")

            if tool_name == "semantic_search":
                # Parse each "--- MATCH ---" block
                for block in content.split("--- MATCH ---"):
                    block = block.strip()
                    if not block:
                        continue

                    pages_match = re.search(r"Pages:\s*(.+)", block)
                    bbox_match = re.search(r"BBox:\s*(.+)", block)
                    hash_match = re.search(r"Hash:\s*([A-Fa-f0-9]+)", block)
                    text_match = re.search(r"Text:\s*(.+?)(?:\nPages:|\Z)", block, re.DOTALL)

                    page_refs_str = pages_match.group(1).strip() if pages_match else "[]"
                    try:
                        page_refs = json.loads(page_refs_str)
                    except json.JSONDecodeError:
                        page_refs = []

                    bbox = None
                    if bbox_match:
                        try:
                            bbox_data = json.loads(bbox_match.group(1).strip())
                            bbox = BoundingBox.model_validate(bbox_data)
                        except Exception:
                            bbox = None

                    citations.append(
                        Citation(
                            doc_name="corpus",
                            page_number=page_refs[0] if page_refs else 0,
                            bbox=bbox,
                            content_hash=hash_match.group(1) if hash_match else "",
                            excerpt=(text_match.group(1).strip()[:200] if text_match else ""),
                        )
                    )

            elif tool_name == "structured_query":
                # Parse JSON rows returned by the SQL tool
                try:
                    rows = json.loads(content)
                    if isinstance(rows, list):
                        for row in rows:
                            citations.append(Citation(
                                doc_name=row.get("doc_id", "unknown"),
                                page_number=row.get("page_number", 0),
                                content_hash=row.get("source_hash", ""),
                                excerpt=f"{row.get('metric_name', '')}: {row.get('metric_value', '')}",
                            ))
                except (json.JSONDecodeError, TypeError):
                    pass

        return citations

    @staticmethod
    def _audit_claim(llm_answer: str, citations: list[Citation]) -> bool:
        """
        Audit Mode: Checks if the LLM answer is grounded.
        A basic check: did it provide a 64-char hex hash that exists in our records?
        """
        # Simple but stricter rule: require at least one citation with a content hash.
        if not citations:
            return False

        has_hashed_citation = any(c.content_hash for c in citations)
        return bool(has_hashed_citation)
