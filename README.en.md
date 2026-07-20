# doc-search-lite

<p align="center">
  <strong>Your personal document deep-research assistant.</strong>
</p>

<div align="center">
  🔍 PDF/DOCX/XLSX/PPTX → Markdown → BM25 Index → LLM-powered Search &nbsp;|&nbsp; 🚫 No vector DB &nbsp;|&nbsp; 🖥️ CLI + Web + API + MCP
</div>

<br>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![CI](https://github.com/rickqi/doc-search-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/rickqi/doc-search-lite/actions/workflows/ci.yml)

---

## 💥 Introduction

**doc-search** started as a personal tool to solve a simple problem:
hundreds of insurance product clauses and medical diagnostic manuals
scattered across folders — impossible to search by keywords alone.

Over 50 releases in two months (v0.1 → v0.21, May–July 2026), it grew into a full-featured
**local document intelligence system** that converts any business document
to Markdown, builds a Tantivy BM25 index, and lets an LLM agent
search, read, cross-reference, and answer questions from your own knowledge base.
No documents ever leave your machine. No vector database required.

**doc-search-lite** is the open-source MIT core of that personal tool,
stripped of enterprise-specific features and internal configuration.
If you have a folder full of PDFs, DOCXs, or spreadsheets and need
to ask questions like *"What's our annual leave policy?"* or
*"Show me all clauses about data protection"* — this is for you.

## Why doc-search-lite?

This is the open-source core of [doc-search](https://github.com/rickqi/doc-search),
stripped of enterprise features and internal data. **No vector database, no local model inference.**

| Feature | doc-search (enterprise) | doc-search-lite (OSS) |
|---------|:----------------------:|:---------------------:|
| BM25 + Agent RAG search | ✅ | ✅ |
| Multi-format document conversion | ✅ | ✅ |
| Web UI + API + MCP | ✅ | ✅ |
| PII desensitization | ✅ | ✅ |
| Hybrid search (BM25+Grep RRF) | ✅ | ✅ |
| Multi-index search | ✅ | ✅ |
| Document structure awareness | ✅ | ✅ |
| CLI stats / diagnostics / budget | ✅ | ✅ |
| PDF enhancement (LA-3B) | ✅ | ❌ |
| Dify external knowledge API | ✅ | ❌ |
| Pi TUI | ✅ (deprecated) | ❌ |
| QA benchmark scripts | ✅ | ❌ |
| OpenCode Skill | ✅ | ❌ |
| License | PolyForm Strict | **MIT** |

### Comparison with DCI-Agent-Lite

[DCI-Agent-Lite](https://github.com/DCI-Agent/DCI-Agent-Lite) is an academic
research framework for the **Direct Corpus Interaction** paradigm —
an agent searches raw text corpora using terminal tools (`rg`, `find`, `sed`)
with no indexing. Both projects share the philosophy of **no vector databases**,
but target different use cases:

| Dimension | doc-search-lite | DCI-Agent-Lite |
|-----------|----------------|----------------|
| **Purpose** | Production document search system | Academic benchmark/evaluation |
| **Corpus** | Your own PDF/DOCX/XLSX/PPTX/HTML | Pre-formatted JSONL datasets (Wikipedia, BrowseComp) |
| **Indexing** | Tantivy BM25 (Rust) + jieba Bigram | **Zero index** — raw `rg`/`find` on text files |
| **Search modes** | BM25 / Grep / Hybrid (RRF) / Tag / Agent | Agent-only (bash tool loop) |
| **Document support** | 11 formats + OCR for images | Plain text / JSONL only |
| **Agent framework** | Custom SearchAgent (COMPILOT P0-P6) | Pi coding agent (bash + context mgmt) |
| **Interface** | CLI + Web UI + REST API + MCP 4 tools | Pi TUI only (`--terminal`) |
| **APIs** | FastAPI (21 routes), FastMCP (4 tools), SSE | None |
| **Observability** | Usage tracking, budget guard, 14-step diagnostics, search logging | None |
| **Security** | PII desensitization, API key auth | None |
| **Target audience** | Teams deploying document search | Researchers benchmarking agentic search |
| **License** | MIT | Apache 2.0 |
| **Paper** | — | [arXiv:2605.05242](https://arxiv.org/abs/2605.05242) |

## Quick Start

```bash
# 1. Install
git clone https://github.com/rickqi/doc-search-lite.git
cd doc-search-lite
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# 2. Copy env config
copy .env.example .env
# Edit .env, set GLM_API_KEY=your-key

# 3. Convert documents
.venv\Scripts\python -m src.cli batch-convert ./docs --raw-root ./raw

# 4. Build search index
.venv\Scripts\python -m src.cli build-index ./raw

# 5. Search
.venv\Scripts\python -m src.cli query "search query" -i ./raw/index --agent

# 6. Launch Web UI
.venv\Scripts\python -m src.api
```

> Requires **Python 3.10+** and a **ZhipuAI GLM API key** (also used for Rerank & OCR).
> DeepSeek is supported as an alternative LLM provider.

## Demo

<p align="center">
  <img src="docs/screenshots/web-ui.png" alt="doc-search-lite Web UI" width="80%">
  <br>
  <em>Web UI — Agent search with SSE streaming, tool call trace, and source citations. <a href="docs/screenshots/">More screenshots →</a></em>
</p>

## Features

- **Multi-format conversion**: PDF, DOCX, XLSX, PPTX, HTML, CSV, TXT, images (OCR), Outlook MSG, ZIP/7z/RAR archives
- **BM25 full-text search**: Tantivy (Rust) + jieba Chinese tokenization + Bigram fallback
- **Hybrid search**: BM25 + Grep parallel with RRF fusion, configurable profiles (legal/technical/faq/general)
- **Multi-index search**: Cross-database search with metadata routing
- **Agentic RAG**: LLM-driven tool loop (search/read/grep/rerank) with dynamic confidence, sufficiency checks, and convergence guards
- **COMPILOT optimizations** (v0.14+): ReAct reasoning, Draft verification loop, Tool feedback signals, Convergence nudging, Best-of-K, Confidence calibration
- **MCP Fast Pipeline** (v0.15+): Query rewriting + multi-query BM25 + speculative pre-read, ~12-18s vs 40-90s full tool loop
- **MCP Server**: FastMCP with 4 tools (`doc_search`/`doc_agent`/`doc_read`/`doc_analyze`), auto-index discovery
- **Dual LLM provider**: ZhipuAI GLM / DeepSeek, one-click switch
- **Tiered Model Routing**: Fast model for intermediate steps, power model for final answer
- **Web UI**: SSE streaming, session management, DB panel with token usage charts, file upload
- **PII desensitization**: Phone/ID/bank card masking before LLM calls, automatic restore
- **Directory watching**: Watchdog auto-indexing on file changes
- **Search modes**: BM25 / Grep / Hybrid / Tag / Agent — CLI + API + MCP
- **Skill system**: 6 built-in analysis skills + external SKILL.md loading
- **Stats & budget**: Usage tracking (millicents), budget guard, search logging, diagnostics (14-step timing)
- **5 complexity levels**: simple(2 rounds) / light(4) / medium(8) / complex(8 + decompose + verify + BOK)

## Architecture

```mermaid
flowchart TB
    classDef user fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    classDef gateway fill:#fef3c7,stroke:#eab308,stroke-width:2px
    classDef agent fill:#d1fae5,stroke:#10b981,stroke-width:2px
    classDef search fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px
    classDef convert fill:#e0e7ff,stroke:#6366f1,stroke-width:2px
    classDef data fill:#fce7f3,stroke:#ec4899,stroke-width:2px
    classDef external fill:#f8fafc,stroke:#94a3b8,stroke-width:2px,stroke-dasharray:6 4
    classDef comp fill:#f3f4f6,stroke:#9ca3af,stroke-width:1px,font-size:9px

    subgraph Users["🎯 User Interface Layer"]
        CLI["CLI (Click)\ndoc-search-lite batch-convert / query"]
        WebUI["Web UI\nFastAPI + vanilla CSS/JS + SSE"]
        MCP["MCP Server\nFastMCP - 4 tools"]
        REST["REST API\n21+ endpoints"]
    end
    subgraph Gateway["⚙️ API & Gateway Layer"]
        Auth["Auth Middleware\nBearer / X-API-Key"]
        Session["Session Manager\nSSE + SQLite persistence"]
        Intent["Intent Classifier\n3-mode routing"]
    end
    subgraph Agent["🧠 Agent Intelligence Layer"]
        SA["SearchAgent\ntool_loop (8 rounds) / pipeline"]
        LLM["LLMClient\nLiteLLM - GLM / DeepSeek"]
        Tools["6 Agent Tools\nSearch / Grep / Read / Rerank / Summarize / Bash"]
        subgraph COMPILOT["COMPILOT Optimizations"]
            direction LR
            P0["P0 ReAct\nThought→Action"]
            P1["P1 Draft Verify\nClosed-loop"]
            P2["P2 Feedback\nTool signals"]
            P3["P3-P6\nNudge / BOK / Confidence"]
        end
    end
    subgraph Search["🔍 Search Pipeline Layer"]
        BM25["BM25\nTantivy + jieba + Bigram"]
        HY["Hybrid\nBM25+Grep RRF K=60"]
        MI["Multi-Index\nFan-out + namespace"]
        RK["Reranker\nZhipuAI / local bge"]
    end
    subgraph Convert["📄 Conversion Pipeline Layer"]
        PDF["PDF\npdfplumber+pypdf+OCR"]
        OFF["Office\nMarkItDown"]
        CSV["HTML/CSV\ntable alignment fix"]
        IMG["Images\nOCR 4 engines"]
        ARC["Archives\nZIP/7z/RAR/tar"]
    end
    subgraph Data["🗄️ Data & Persistence Layer"]
        TIDX["Tantivy Index\nBM25 - title boost + highlights"]
        CDB["ConvertDB (SQLite)\nSchema v2.1"]
        FS["File Store\n.md + .md.json"]
    end
    subgraph External["☁️ External Services"]
        GLM["ZhipuAI GLM\nLLM + Rerank + OCR"]
        DS["DeepSeek\nAlternative LLM"]
        POCR["PaddleOCR\nLocal GPU OCR"]
        LIT["LiteLLM\n200+ provider proxy"]
    end
    class CLI,WebUI,MCP,REST user
    class Auth,Session,Intent gateway
    class SA,LLM,Tools agent
    class BM25,HY,MI,RK search
    class PDF,OFF,CSV,IMG,ARC convert
    class TIDX,CDB,FS data
    class GLM,DS,POCR,LIT external
    class P0,P1,P2,P3 comp

    CLI --> Auth
    WebUI --> Auth
    MCP --> Auth
    REST --> Auth
    Auth --> Session
    Session --> Intent
    Intent --> SA
    SA --> Tools
    SA --> COMPILOT
    Tools --> BM25
    Tools --> HY
    Tools --> MI
    BM25 --> RK
    HY --> RK
    MI --> RK
    BM25 --> TIDX
    HY --> TIDX
    MI --> TIDX
    PDF --> FS
    OFF --> FS
    CSV --> FS
    IMG --> FS
    ARC --> FS
    FS --> TIDX
    GLM --> LLM
    DS --> LLM
```

### Pipeline

### Pipeline

```
Documents (PDF/DOCX/XLSX/PPTX/HTML/CSV/TXT/Images)
    │
    ConverterCoordinator → Markdown → .md + .md.json (headings, tags)
    │
    Tantivy Index (jieba + Bigram + title boost)
    │
    ┌── BM25 keyword search ────┐
    ├── Grep regex search        ├── 4 modes
    ├── Hybrid RRF fusion        │
    └── Tag-based recall ────────┘
    │
    ┌── Agent tool_loop (8 rounds) ──┐
    │  search → read → search →     │  COMPILOT P0-P6
    │  read → rerank → synthesize   │
    └────────────────────────────────┘
    │
    LLM (GLM / DeepSeek) → Answer with citations
```

### Local Database (convert.db)

Each raw directory gets a `convert.db` (SQLite, WAL mode) that tracks every file's lifecycle end-to-end:

```
convert.db (per raw/ directory)
├── Schema: "2.1" (auto-migrated from 1.1 → 2.0 → 2.1)
├── WAL mode, foreign keys enabled
│
├── directories/     # Directory tree mirroring source structure
├── files/           # Per-file state machine
│   ├── status: pending → converting → success | failed | skipped
│   ├── source_hash, mtime for incremental detection
│   ├── converter, convert_time, ocr_tokens, pipeline_version
│   └── metadata_json, last_error
├── batches/         # Conversion batch history (resume support)
├── skipped/         # Skip reasons (unsupported format, password-protected)
├── config/          # Schema version, pipeline metadata
│
├── token_usage/     # OCR/LLM token consumption (per-file, per-model)
├── pricing/         # Model price mapping (millicents per token)
├── budget/          # Monthly/total budget limits and spending
│
├── search_feedback/ # 👍/👎 user relevance feedback
├── auth_log/        # API authentication audit trail
│
├── query_diagnostics/  # 14-step query performance timing
└── llm_call_log/       # Per-call LLM latency, tokens, retry count
```

## Commands

### Document Conversion

```bash
python -m src.cli batch-convert ./docs --raw-root ./raw
python -m src.cli batch-convert ./docs --raw-root ./raw --mode incremental
python -m src.cli batch-convert ./docs --raw-root ./raw --parallel 4
python -m src.cli batch-convert ./docs --raw-root ./raw --force
python -m src.cli batch-convert ./docs --raw-root ./raw --no-ocr
```

### Index Management

```bash
python -m src.cli build-index ./raw
python -m src.cli watch ./raw --debounce 1.0
python -m src.cli build-index ./raw --chunk-mode
```

### Search

```bash
python -m src.cli query "annual leave policy" -i ./raw/index -l 5
python -m src.cli query "confidentiality" -i ./raw
python -m src.cli query "data protection" -i ./raw/index --search-mode hybrid
python -m src.cli query "报销" -i ./raw/index --search-mode tag
python -m src.cli query "keyword" -i ./raw/index --export json -o results.json
```

### Agent Search

```bash
python -m src.cli query "How do I apply for annual leave?" -i ./raw/index --agent
python -m src.cli query "What's the travel reimbursement policy?" -i ./raw/index --agent --rerank
python -m src.cli query "出差标准" -i ./raw/index --agent --skill summarize
python -m src.cli query "" -i ./raw/index --interactive
```

### Web UI

```bash
python -m src.api
python -m src.api --host 0.0.0.0 --port 8080
```

### MCP Server

```bash
pip install -e ".[mcp]"
python -m src.mcp_server
```

**MCP Tools**:

| Tool | Description |
|------|-------------|
| `doc_search` | BM25 / Hybrid / Grep keyword search |
| `doc_agent` | Agentic RAG with LLM answer generation |
| `doc_read` | Read full document content by doc_id or source_path |
| `doc_analyze` | Deep document analysis (compare/extract/summarize/table) |

### Stats & Diagnostics

```bash
python -m src.cli stats summary --days 7
python -m src.cli stats daily --days 30
python -m src.cli stats export --format html -o report.html
python -m src.cli stats budget list
python -m src.cli stats diagnostics --days 7
python -m src.cli stats slow-queries --threshold 30000
```

### Directory Migration

```bash
python -m src.cli diff-migrate /path/to/base /path/to/compare
python -m src.cli diff-migrate /path/to/base /path/to/compare --export-new /path/to/export
```

## Supported Formats

| Format | Extension | Converter |
|--------|-----------|-----------|
| PDF | `.pdf` | pdfplumber + pypdf (scanned PDF auto OCR) |
| Word | `.docx` | MarkItDown |
| Excel | `.xlsx`, `.xls` | MarkItDown (>5MB auto LibreOffice → CSV) |
| PowerPoint | `.pptx` | MarkItDown |
| HTML | `.html`, `.htm` | MarkItDown + table alignment fix |
| CSV | `.csv` | pandas + auto encoding detection |
| Text | `.txt` | Auto encoding (utf-8/gbk/gb2312) |
| Markdown | `.md` | Pass-through |
| Images | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp` | ZhipuAI / PaddleOCR / PP-StructureV3 |
| Email | `.msg` | olefile (Outlook OLE2) |
| Archives | `.zip`, `.7z`, `.rar`, `.tar`, `.gz` | Extract → convert → clean |

> `.doc` format requires pre-conversion to `.docx` via LibreOffice.

## Configuration

Copy `.env.example` to `.env` and configure:

```ini
GLM_API_KEY=your-glm-api-key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_PROVIDER=glm
LLM_MODEL=glm-4
DEEPSEEK_API_KEY=your-deepseek-api-key
LLM_TIERED_ROUTING=false
WEB_API_KEY=your-secret-key
OCR_ENGINE=zhipu
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Search engine | Tantivy (Rust, Python bindings) |
| Chinese tokenization | jieba + Bigram fallback |
| LLM integration | LiteLLM (GLM / DeepSeek / 200+ providers) |
| Rerank | ZhipuAI cloud API (default) or local bge-reranker-v2-m3 |
| OCR | ZhipuAI / PaddleOCR / PaddleOCR HTTP / PP-StructureV3 |
| Document conversion | MarkItDown 0.1.x, pdfplumber, pypdf, olefile, pandas |
| Web framework | FastAPI + SSE + vanilla CSS/JS + Chart.js |
| CLI framework | Click + Rich |
| Storage | SQLite (WAL mode), Tantivy index, filesystem |
| File watching | watchdog |

## Key Design Decisions

- **No vector database**: BM25 + jieba provides better keyword precision for legal/insurance/regulatory documents
- **Whole-document indexing**: Preserves full context vs chunk-splitting that loses document structure
- **Result-based error handling**: `ConvertResult(success, errors)` and `ToolResult.ok()/.fail()` throughout
- **Optional traceability**: `UsageTracker=None` everywhere — zero cost when not configured
- **Fail-safe desensitization**: PII masking failures fall back to original text, never block LLM calls
- **Tiered routing**: Fast cheap model for intermediate steps, expensive model only for final answer

## Development

```bash
.venv\Scripts\python.exe -m pytest tests/ -q --tb=short
.venv\Scripts\python.exe -m pytest tests/ --cov
.venv\Scripts\ruff check src/ tests/
.venv\Scripts\ruff format src/ tests/
```

## License

MIT License — see [LICENSE](LICENSE).
