# 🏭 Document Intelligence Refinery

A production-grade, multi-stage agentic pipeline that ingests heterogeneous document corpora (PDFs, scanned reports, table-heavy documents) and emits structured, queryable, spatially-indexed knowledge.

**TRP1 Challenge — Week 3 | FDE Program**

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- (Optional) [Ollama](https://ollama.com/) for local LLM — used by Stages 4 & 5
- (Optional) API key for OpenAI / Gemini — for cloud LLM or Vision extraction

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd trp1-document-intelligence-refinery

# Install dependencies
uv sync

# Install dev dependencies (for testing)
uv sync --extra dev

# Configure environment
cp .env.example .env
# Edit .env with your preferred LLM provider and API keys
```

### Environment Variables

| Variable               | Default                        | Description                                               |
| :--------------------- | :----------------------------- | :-------------------------------------------------------- |
| `LLM_PROVIDER`         | `ollama`                       | LLM provider for Stage 4–5 (`ollama`, `openai`, `gemini`) |
| `LLM_MODEL`            | `qwen2.5:latest`               | Model for summaries, fact extraction, and query agent     |
| `VLM_PROVIDER`         | `ollama`                       | Vision LLM for Strategy C (`ollama`, `openai`, `gemini`)  |
| `VLM_MODEL`            | `llava`                        | Vision model for scanned document extraction              |
| `OLLAMA_BASE_URL`      | `http://localhost:11434`       | Ollama server URL                                         |
| `OPENAI_API_KEY`       | —                              | Required if using OpenAI as provider                      |
| `GOOGLE_API_KEY`       | —                              | Required if using Gemini as provider                      |
| `OPENROUTER_API_KEY`   | —                              | Alternative API gateway for Vision extraction             |
| `REFINERY_CONFIG_PATH` | `rubric/extraction_rules.yaml` | Path to the configuration file                            |

---

## Usage

### Run Full Pipeline (All 5 Stages)

```bash
# Single document
uv run python -m src.main --input corpus/tax_expenditure_ethiopia_2021_22.pdf

# Entire corpus directory
uv run python -m src.main --input corpus/
```

The pipeline runs: **Triage → Extraction → Chunking → PageIndex → VectorStore** for each document.

### Run Triage Only (Stage 1)

```bash
uv run python -m src.main --input corpus/ --triage-only
```

### Run Query Evaluation (Stage 5)

After processing documents through the pipeline, generate the Q&A evaluation:

```bash
uv run python -m src.generate_evals
```

This runs 12 pre-defined queries (3 per document class) against the LangGraph Query Agent and outputs `docs/evaluation_qa.md`.

### Run Tests

```bash
uv run pytest tests/ -v
```

### Docker

```bash
docker build -t refinery .
docker run -v ./corpus:/app/corpus -v ./.refinery:/app/.refinery refinery --input corpus/tax_expenditure_ethiopia_2021_22.pdf
```

---

## Project Structure

```
trp1-document-intelligence-refinery/
├── src/
│   ├── main.py                          # CLI entrypoint — orchestrates all 5 stages
│   ├── config.py                        # YAML config loader
│   ├── generate_evals.py               # Q&A evaluation generator (12 queries)
│   ├── models/
│   │   └── schemas.py                   # All Pydantic models (DocumentProfile, ExtractedDocument, LDU, PageIndexNode, ProvenanceChain, etc.)
│   ├── agents/
│   │   ├── triage.py                    # Stage 1: Document Triage Agent
│   │   ├── extractor.py                # Stage 2: Extraction Router (confidence-gated)
│   │   ├── chunker.py                  # Stage 3: Semantic Chunking Engine + ChunkValidator
│   │   ├── indexer.py                  # Stage 4: PageIndex Builder + FactTable Extractor
│   │   └── query_agent.py             # Stage 5: LangGraph ReAct Agent (3 tools)
│   ├── strategies/
│   │   ├── base.py                      # Strategy ABC interface
│   │   ├── fast_text.py                 # Strategy A: pdfplumber (Free, <0.1s/page)
│   │   ├── layout_extractor.py          # Strategy B: Docling (Free local, 1-5s/page)
│   │   └── vision_extractor.py          # Strategy C: VLM (OpenRouter/Gemini, ~$0.001/page)
│   ├── data/
│   │   ├── vector_store.py             # ChromaDB wrapper for LDU embeddings
│   │   └── fact_store.py               # SQLite FactTable for numerical metrics
│   └── utils/
│       └── llm.py                       # Dynamic LLM client (Ollama/OpenAI/Gemini)
├── tests/
│   ├── test_triage.py                   # Triage Agent classification tests
│   └── test_confidence.py              # Confidence scoring tests
├── rubric/
│   └── extraction_rules.yaml           # Externalized thresholds, chunking rules, budget guard
├── corpus/                              # Input documents (50 PDFs, 4 document classes)
├── .refinery/                           # Pipeline output artifacts
│   ├── profiles/                        # DocumentProfile JSONs (1 per document)
│   ├── extraction_ledger.jsonl         # Extraction audit trail (strategy, confidence, cost)
│   ├── pageindex/                       # PageIndex trees (JSON, 1 per document)
│   ├── chroma_db/                       # ChromaDB vector store (LDU embeddings)
│   └── facts.db                         # SQLite FactTable (financial/numerical metrics)
├── docs/
│   ├── final_report.md                  # Unified final submission report
│   └── evaluation_qa.md               # 12 Q&A examples with provenance (3 per class)
├── DOMAIN_NOTES.md                      # Phase 0 domain onboarding (708 lines)
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Architecture — The 5-Stage Refinery Pipeline

```
┌─────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────┐    ┌─────────────┐
│ Stage 1  │───▶│   Stage 2    │───▶│   Stage 3     │───▶│  Stage 4  │───▶│   Stage 5   │
│ Triage   │    │  Extraction  │    │  Chunking     │    │ PageIndex │    │ Query Agent │
│ Agent    │    │  Router      │    │  Engine       │    │ Builder   │    │ (LangGraph) │
└─────────┘    └──────────────┘    └───────────────┘    └───────────┘    └─────────────┘
     │               │                    │                   │                │
     ▼          ┌────┴─────┐              ▼              ┌───┴───┐      ┌────┴────┐
DocumentProfile  A    B     C         List[LDU]        PageIndex   │    3 Tools   │
               fast layout vision  + ChunkValidator    + FactTable  │  navigate   │
               text aware  VLM                                     │  search     │
                                                                   │  sql_query  │
                                                                   └─────────────┘
```

### Extraction Strategy Routing

| Strategy             | Tool             | Cost         | Triggers When                            |
| -------------------- | ---------------- | ------------ | ---------------------------------------- |
| **A — Fast Text**    | pdfplumber       | Free         | `native_digital` + `single_column`       |
| **B — Layout-Aware** | Docling          | Free (local) | `multi_column` / `table_heavy` / `mixed` |
| **C — Vision**       | VLM (OpenRouter) | ~$0.001/page | `scanned_image` / low confidence         |

### Confidence-Gated Escalation Guard

```
Strategy A → confidence < 0.70 → Escalate to Strategy B
Strategy B → confidence < 0.45 → Escalate to Strategy C
```

The escalation guard operates at **page-level**, not document-level. This avoids wasting Vision API budget on clean pages within mixed documents.

---

## Configuration

All thresholds and parameters are externalized in `rubric/extraction_rules.yaml`:

- **Confidence gates**: Strategy A/B minimum thresholds
- **Page signal thresholds**: char count, density, image ratio
- **Chunking constitution**: 5 enforced rules, max tokens, overlap
- **Budget guard**: max USD per document/page
- **Domain keywords**: financial, legal, technical, medical

A new document type can be onboarded by modifying only `extraction_rules.yaml` — no code changes required.

Override the config path:

```bash
export REFINERY_CONFIG_PATH=path/to/custom/extraction_rules.yaml
```

---

## Key Design Decisions

1. **Page-level escalation** over document-level — saves Vision API budget on mixed documents
2. **Multi-signal confidence formula** (5 signals) — prevents ghost text layer false positives
3. **Docling over MinerU** for Strategy B — typed `DoclingDocument` maps cleanly to Pydantic schemas
4. **PageIndex for long documents** — narrows vector search to relevant sections first
5. **All thresholds in YAML** — enables FDE-ready onboarding without code changes
