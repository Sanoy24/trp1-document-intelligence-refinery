# 🏭 Document Intelligence Refinery

A production-grade, multi-stage agentic pipeline that ingests heterogeneous document corpora (PDFs, scanned reports, table-heavy documents) and emits structured, queryable, spatially-indexed knowledge.

**TRP1 Challenge — Week 3 | FDE Program**

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd trp1-document-intelligence-refinery

# Install dependencies
uv sync

# Install dev dependencies (for testing)
uv sync --extra dev
```

### Environment Variables (optional, for Vision extraction)

```bash
export OPENROUTER_API_KEY="your-key-here"   # For Strategy C (VLM)
# OR
export GEMINI_API_KEY="your-key-here"
```

---

## Usage

### Run Triage Only (Stage 1)

```bash
# Profile a single document
uv run python -m src.main --input corpus/tax_expenditure_ethiopia_2021_22.pdf --triage-only

# Profile all corpus documents
uv run python -m src.main --input corpus/ --triage-only
```

### Run Full Pipeline (Triage + Extraction)

```bash
# Single document
uv run python -m src.main --input corpus/tax_expenditure_ethiopia_2021_22.pdf

# Entire corpus
uv run python -m src.main --input corpus/
```

### Run Tests

```bash
uv run pytest tests/ -v
```

---

## Project Structure

```
trp1-document-intelligence-refinery/
├── src/
│   ├── main.py                          # CLI entrypoint
│   ├── config.py                        # YAML config loader
│   ├── models/
│   │   └── schemas.py                   # All Pydantic models
│   ├── agents/
│   │   ├── triage.py                    # Stage 1: Document Triage Agent
│   │   └── extractor.py                # Stage 2: Extraction Router
│   └── strategies/
│       ├── base.py                      # Strategy ABC
│       ├── fast_text.py                 # Strategy A: pdfplumber (Low cost)
│       ├── layout_extractor.py          # Strategy B: Docling (Medium cost)
│       └── vision_extractor.py          # Strategy C: VLM (High cost)
├── tests/
│   ├── test_triage.py                   # Triage Agent classification tests
│   └── test_confidence.py              # Confidence scoring tests
├── rubric/
│   └── extraction_rules.yaml           # Chunking constitution & thresholds
├── corpus/                              # Input documents (4 classes)
├── .refinery/
│   ├── profiles/                        # DocumentProfile JSONs
│   └── extraction_ledger.jsonl         # Extraction audit trail
├── reports/                             # Interim & final reports
├── DOMAIN_NOTES.md                      # Phase 0 domain onboarding
├── pyproject.toml
└── README.md
```

---

## Architecture — The 5-Stage Refinery Pipeline

```
┌─────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────┐    ┌─────────────┐
│ Stage 1  │───▶│   Stage 2    │───▶│   Stage 3     │───▶│  Stage 4  │───▶│   Stage 5   │
│ Triage   │    │  Extraction  │    │  Chunking     │    │ PageIndex │    │ Query Agent │
│ Agent    │    │  Router      │    │  Engine       │    │ Builder   │    │             │
└─────────┘    └──────────────┘    └───────────────┘    └───────────┘    └─────────────┘
     │               │
     │          ┌────┴─────┐
     ▼          ▼    ▼     ▼
DocumentProfile  A    B     C
                fast layout vision
                text aware  VLM
```

### Extraction Strategy Routing

| Strategy             | Tool             | Cost         | Triggers When                            |
| -------------------- | ---------------- | ------------ | ---------------------------------------- |
| **A — Fast Text**    | pdfplumber       | Free         | `native_digital` + `single_column`       |
| **B — Layout-Aware** | Docling          | Free (local) | `multi_column` / `table_heavy` / `mixed` |
| **C — Vision**       | VLM (OpenRouter) | ~$0.001/page | `scanned_image` / low confidence         |

### Escalation Guard

```
Strategy A → confidence < 0.70 → Strategy B → confidence < 0.45 → Strategy C
```

---

## Configuration

All thresholds and parameters are externalized in `rubric/extraction_rules.yaml`:

- **Confidence gates**: Strategy A/B minimum thresholds
- **Page signal thresholds**: char count, density, image ratio
- **Chunking constitution**: max tokens, overlap, rules
- **Budget guard**: max USD per document/page
- **Domain keywords**: financial, legal, technical, medical

Override the config path:

```bash
export REFINERY_CONFIG_PATH=path/to/custom/extraction_rules.yaml
```
