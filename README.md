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

<!--
  TODO: Add screenshots:
  1. docs/screenshots/web-ui.png    — Web UI with a completed Agent search
  2. docs/screenshots/cli-demo.gif   — Terminal recording of CLI workflow
  3. docs/screenshots/db-panel.png  — DB stats and token usage chart
  4. docs/screenshots/mcp.png       — MCP tools in OpenCode/Claude
-->

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

<div style="width: 100%; max-width: 1200px; box-sizing: border-box; position: relative; background: #fafbff; padding: 20px; border-radius: 10px;">
<style scoped>
.arch-wrapper{display:flex;gap:12px}.arch-sidebar{width:165px;flex-shrink:0}.arch-main{flex:1;min-width:0}.arch-title{text-align:center;font-size:22px;font-weight:bold;color:#1e293b;margin-bottom:16px}
.arch-layer{margin:8px 0;padding:14px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.04)}.arch-layer-title{font-size:13px;font-weight:bold;margin-bottom:10px;text-align:center}
.arch-grid{display:grid;gap:8px}.arch-grid-2{grid-template-columns:repeat(2,1fr)}.arch-grid-3{grid-template-columns:repeat(3,1fr)}.arch-grid-4{grid-template-columns:repeat(4,1fr)}.arch-grid-5{grid-template-columns:repeat(5,1fr)}
.arch-box{border-radius:6px;padding:8px;text-align:center;font-size:11px;font-weight:600;line-height:1.35;color:#1e293b;background:#fff;border:1px solid #e2e8f0}.arch-box.highlight{background:linear-gradient(135deg,#dbeafe 0%,#bfdbfe 100%);border:2px solid #2563eb}.arch-box.tech{font-size:10px;color:#475569;background:#f8fafc}
.arch-layer.external{background:linear-gradient(135deg,#f8fafc 0%,#f1f5f9 100%);border:2px dashed #94a3b8}.arch-layer.external .arch-layer-title{color:#64748b}
.arch-layer.user{background:linear-gradient(135deg,#dbeafe 0%,#bfdbfe 100%);border:2px solid #3b82f6}.arch-layer.user .arch-layer-title{color:#1e40af}
.arch-layer.application{background:linear-gradient(135deg,#fef3c7 0%,#fde68a 100%);border:2px solid #eab308}.arch-layer.application .arch-layer-title{color:#854d0e}
.arch-layer.ai{background:linear-gradient(135deg,#d1fae5 0%,#a7f3d0 100%);border:2px solid #10b981}.arch-layer.ai .arch-layer-title{color:#065f46}
.arch-layer.data{background:linear-gradient(135deg,#fce7f3 0%,#fbcfe8 100%);border:2px solid #ec4899}.arch-layer.data .arch-layer-title{color:#9d174d}
.arch-layer.search{background:linear-gradient(135deg,#ede9fe 0%,#ddd6fe 100%);border:2px solid #8b5cf6}.arch-layer.search .arch-layer-title{color:#5b21b6}
.arch-layer.convert{background:linear-gradient(135deg,#e0e7ff 0%,#c7d2fe 100%);border:2px solid #6366f1}.arch-layer.convert .arch-layer-title{color:#4338ca}
.arch-sidebar-panel{border-radius:8px;padding:10px;background:linear-gradient(135deg,#f3f4f6 0%,#e5e7eb 100%);border:1px solid #9ca3af;margin-bottom:8px}.arch-sidebar-title{font-size:12px;font-weight:bold;text-align:center;color:#1e293b;margin-bottom:6px}.arch-sidebar-item{font-size:10px;text-align:center;color:#374151;background:#fff;padding:5px;border-radius:5px;margin:3px 0;border:1px solid #e5e7eb}.arch-sidebar-item.metric{background:#dbeafe;border:1px solid #3b82f6;color:#1e40af;font-weight:600}
.arch-conn{stroke:#94a3b8;stroke-width:1.5;fill:none}.arch-conn-dashed{stroke:#94a3b8;stroke-width:1.5;fill:none;stroke-dasharray:6 4}.arch-conn-label{font-size:9px;fill:#64748b;font-family:sans-serif}
</style>
<div class="arch-title">doc-search-lite — System Architecture</div>
<div class="arch-wrapper">
<div class="arch-sidebar">
<div class="arch-sidebar-panel"><div class="arch-sidebar-title">🖥️ DevOps</div><div class="arch-sidebar-item">CLI (Click)<br><small>batch-convert<br>build-index<br>query / watch / stats</small></div><div class="arch-sidebar-item">pip install<br><small>Python 3.10+</small></div><div class="arch-sidebar-item">.env Config<br><small>GLM/DeepSeek Keys</small></div></div>
<div class="arch-sidebar-panel"><div class="arch-sidebar-title">📊 Observability</div><div class="arch-sidebar-item metric">UsageTracker<br><small>OCR/LLM/Rerank</small></div><div class="arch-sidebar-item metric">BudgetGuard<br><small>Monthly Caps</small></div><div class="arch-sidebar-item metric">Diagnostics<br><small>14-Step Timing</small></div><div class="arch-sidebar-item metric">SearchLogger<br><small>Async Logging</small></div><div class="arch-sidebar-item">AgentMemory<br><small>Q&A Recall</small></div></div>
<div class="arch-sidebar-panel"><div class="arch-sidebar-title">📦 Storage</div><div class="arch-sidebar-item">Tantivy Index<br><small>BM25 Schema v2</small></div><div class="arch-sidebar-item">ConvertDB<br><small>SQLite Schema v2.1</small></div><div class="arch-sidebar-item">Markdown Files<br><small>.md + .md.json</small></div></div>
</div>
<div class="arch-main">
<div class="arch-layer user">
<div class="arch-layer-title">🎯 User Interface Layer</div>
<div class="arch-grid arch-grid-4">
<div class="arch-box highlight">CLI (Click)<br><small>doc-search-lite batch-convert / query</small></div>
<div class="arch-box highlight">Web UI<br><small>FastAPI + vanilla HTML/CSS/JS + SSE</small></div>
<div class="arch-box">MCP Server<br><small>FastMCP — 4 tools for OpenCode/Claude</small></div>
<div class="arch-box">REST API<br><small>21+ endpoints + Swagger docs</small></div>
</div>
</div>
<div class="arch-layer application">
<div class="arch-layer-title">⚙️ API & Gateway Layer</div>
<div class="arch-grid arch-grid-3">
<div class="arch-box">Auth Middleware<br><small>Bearer / X-API-Key / Token Store</small></div>
<div class="arch-box">Session Manager<br><small>SSE streaming + SQLite persistence</small></div>
<div class="arch-box">Intent Classifier<br><small>search / review / direct 3-mode routing</small></div>
</div>
</div>
<div class="arch-layer ai">
<div class="arch-layer-title">🧠 Agent Intelligence Layer</div>
<div class="arch-grid arch-grid-3">
<div class="arch-box highlight">SearchAgent<br><small>tool_loop (8 rounds) / pipeline</small></div>
<div class="arch-box">LLMClient<br><small>LiteLLM — GLM / DeepSeek</small></div>
<div class="arch-box">Agent Tools<br><small>SearchTool / GrepTool / ReadTool<br>RerankTool / SummarizeTool / BashTool</small></div>
</div>
<div style="margin-top:8px"><div class="arch-grid arch-grid-4">
<div class="arch-box tech">P0 ReAct<br><small>Thought→Action</small></div>
<div class="arch-box tech">P1 Draft Verify<br><small>Closed-loop</small></div>
<div class="arch-box tech">P2 Feedback<br><small>Tool signals</small></div>
<div class="arch-box tech">P3-P6<br><small>Nudge/BOK/Confidence</small></div>
</div></div>
</div>
<div class="arch-layer search">
<div class="arch-layer-title">🔍 Search Pipeline Layer</div>
<div class="arch-grid arch-grid-4">
<div class="arch-box">BM25<br><small>Tantivy + jieba + Bigram</small></div>
<div class="arch-box">Hybrid<br><small>BM25+Grep RRF K=60</small></div>
<div class="arch-box">Multi-Index<br><small>Fan-out + namespace</small></div>
<div class="arch-box">Reranker<br><small>ZhipuAI cloud / local bge</small></div>
</div>
</div>
<div class="arch-layer convert">
<div class="arch-layer-title">📄 Conversion Pipeline Layer</div>
<div class="arch-grid arch-grid-5">
<div class="arch-box tech">PDF<br><small>pdfplumber+pypdf+OCR</small></div>
<div class="arch-box tech">Office<br><small>MarkItDown (docx/pptx/xlsx)</small></div>
<div class="arch-box tech">HTML/CSV<br><small>table alignment fix</small></div>
<div class="arch-box tech">Images<br><small>OCR (4 engines)</small></div>
<div class="arch-box tech">Archives<br><small>ZIP/7z/RAR/tar</small></div>
</div>
</div>
<div class="arch-layer data">
<div class="arch-layer-title">🗄️ Data & Persistence Layer</div>
<div class="arch-grid arch-grid-3">
<div class="arch-box">Tantivy Index<br><small>BM25 — title boost + highlights</small></div>
<div class="arch-box">ConvertDB (SQLite)<br><small>Schema v2.1 — files/batches/tokens/budget</small></div>
<div class="arch-box">File Store<br><small>.md + .md.json (headings + tags)</small></div>
</div>
</div>
<div class="arch-layer external">
<div class="arch-layer-title">☁️ External Services</div>
<div class="arch-grid arch-grid-4">
<div class="arch-box tech">ZhipuAI GLM<br><small>LLM + Rerank + OCR</small></div>
<div class="arch-box tech">DeepSeek<br><small>Alternative LLM</small></div>
<div class="arch-box tech">PaddleOCR<br><small>Local GPU OCR</small></div>
<div class="arch-box tech">LiteLLM<br><small>200+ provider proxy</small></div>
</div>
</div>
</div>
<div class="arch-sidebar">
<div class="arch-sidebar-panel"><div class="arch-sidebar-title">🔒 Security</div><div class="arch-sidebar-item">PII Desensitizer<br><small>Phone/ID/Bank card masking</small></div><div class="arch-sidebar-item">API Key Auth<br><small>Bearer / X-API-Key</small></div><div class="arch-sidebar-item">Auth Audit Log<br><small>token/endpoint/IP/status</small></div><div class="arch-sidebar-item">Fail-safe Design<br><small>Never block on error</small></div></div>
<div class="arch-sidebar-panel"><div class="arch-sidebar-title">📈 Cost & Budget</div><div class="arch-sidebar-item metric">Tiered Routing<br><small>Flash $0.005/query</small></div><div class="arch-sidebar-item metric">Token Tracking<br><small>Millicents precision</small></div><div class="arch-sidebar-item metric">Budget Limits<br><small>Monthly/total caps</small></div><div class="arch-sidebar-item metric">AlertManager<br><small>Webhook + dedup 5min</small></div></div>
<div class="arch-sidebar-panel"><div class="arch-sidebar-title">🔄 Lifecycle</div><div class="arch-sidebar-item">File Status<br><small>pending→converting→success</small></div><div class="arch-sidebar-item">Batch Resume<br><small>Interrupted→restart</small></div><div class="arch-sidebar-item">Incremental<br><small>Hash + mtime detect</small></div><div class="arch-sidebar-item">Watchdog<br><small>Auto re-index</small></div></div>
</div>
</div>
</div>

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

**File lifecycle**: `pending → converting → success | failed | skipped`. On startup, interrupted (`running`) batches auto-reset to `interrupted` for clean resume.

**Uses**:
- **CLI** `batch-convert`: Resume broken conversions, skip unchanged files
- **CLI** `build-index`: Read file paths from DB
- **CLI** `stats`: Token usage, budget, diagnostics reports
- **Web UI** DB panel: Conversion stats, file list, token chart
- **CLI** `catalog`: List failed/pending files, reindex
- **API** `/api/db/*`: Stats, files, batches, token endpoints
- **Stats** `UsageTracker`/`BudgetGuard`: Record and enforce costs

## Commands

### Document Conversion

```bash
# Batch convert
python -m src.cli batch-convert ./docs --raw-root ./raw

# Incremental (only new/changed files)
python -m src.cli batch-convert ./docs --raw-root ./raw --mode incremental

# Parallel processing
python -m src.cli batch-convert ./docs --raw-root ./raw --parallel 4

# Force re-convert
python -m src.cli batch-convert ./docs --raw-root ./raw --force

# Disable OCR
python -m src.cli batch-convert ./docs --raw-root ./raw --no-ocr

# Dry run (show what would be converted)
python -m src.cli batch-convert ./docs --raw-root ./raw --dry-run
```

### Index Management

```bash
# Build search index
python -m src.cli build-index ./raw

# Watch for changes and auto-update
python -m src.cli watch ./raw --debounce 1.0

# Build with chunk mode (split long docs by headings)
python -m src.cli build-index ./raw --chunk-mode
```

### Search

```bash
# BM25 keyword search
python -m src.cli query "年假" -i ./raw/index -l 5

# Grep search (no index needed, searches .md files directly)
python -m src.cli query "个人信息保护" -i ./raw

# Hybrid search (BM25 + Grep RRF fusion)
python -m src.cli query "个人信息保护" -i ./raw/index --search-mode hybrid

# Tag-based recall
python -m src.cli query "报销" -i ./raw/index --search-mode tag

# Multi-index search (comma-separated paths)
python -m src.cli query "数据安全" -i "idx1,idx2,idx3" --search-mode hybrid

# Export results
python -m src.cli query "关键词" -i ./raw/index --export json -o results.json
python -m src.cli query "关键词" -i ./raw/index --export csv -o results.csv
```

### Agent Search

```bash
# Agent Q&A (LLM autonomously searches + reads + answers)
python -m src.cli query "年假如何申请" -i ./raw/index --agent

# Agent + Rerank
python -m src.cli query "差旅报销标准" -i ./raw/index --agent --rerank

# Agent + built-in skill
python -m src.cli query "年假制度" -i ./raw/index --agent --skill summarize
python -m src.cli query "差旅标准" -i ./raw/index --agent --skill compare
python -m src.cli query "报销流程" -i ./raw/index --agent --skill extract-table
python -m src.cli query "合同审批" -i ./raw/index --agent --skill detailed
python -m src.cli query "制度变更" -i ./raw/index --agent --skill timeline
python -m src.cli query "项目报告" -i ./raw/index --agent --skill action-items

# Agent + external custom skill file
python -m src.cli query "数据安全" -i ./raw/index --agent --load-skill ./my-skill.md

# Interactive mode
python -m src.cli query "" -i ./raw/index --interactive
```

### Web UI

```bash
# Start API server (serves Web UI at http://127.0.0.1:8000)
python -m src.api

# Specify host and port
python -m src.api --host 0.0.0.0 --port 8080
```

The Web UI provides:
- **Chat panel**: SSE streaming, session management, skill selector, search mode selector
- **DB panel**: Conversion stats, file list, token usage chart (Chart.js)
- **File upload**: Drag-and-drop → auto convert → index
- **Auth**: API token / Bearer token support

### MCP Server

```bash
# Install MCP dependency
pip install -e ".[mcp]"

# Start MCP server (stdio transport for OpenCode / Claude)
python -m src.mcp_server

# Configure in opencode.json:
# {
#   "mcp": {
#     "doc_search": {
#       "type": "local",
#       "command": [".venv\\Scripts\\python.exe", "-m", "src.mcp_server"]
#     }
#   }
# }
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
# Usage summary
python -m src.cli stats summary
python -m src.cli stats summary --days 7

# Daily trends
python -m src.cli stats daily --days 30

# By model breakdown
python -m src.cli stats models

# Export report
python -m src.cli stats export --format json -o report.json
python -m src.cli stats export --format csv -o report.csv
python -m src.cli stats export --format html -o report.html

# Budget management
python -m src.cli stats budget list
python -m src.cli stats budget set --name default --limit 10000 --period monthly
python -m src.cli stats budget check

# Real-time monitoring
python -m src.cli stats realtime --interval 5

# Performance diagnostics (14-step timing)
python -m src.cli stats diagnostics --days 7
python -m src.cli stats slow-queries --threshold 30000
python -m src.cli stats step-breakdown --days 7
python -m src.cli stats llm-calls --days 7
```

### Directory Migration

```bash
# Compare two directories by content hash
python -m src.cli diff-migrate /path/to/base /path/to/compare

# Export new/changed files
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
# Required
GLM_API_KEY=your-glm-api-key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# LLM Provider (glm or deepseek)
LLM_PROVIDER=glm
LLM_MODEL=glm-4

# Optional: DeepSeek
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Tiered Routing (fast model for intermediate steps)
LLM_TIERED_ROUTING=false
LLM_FAST_MODEL=deepseek-v4-flash
LLM_POWER_MODEL=deepseek-v4-pro

# Web authentication
WEB_API_KEY=your-secret-key

# OCR engine: zhipu | paddleocr | paddleocr-http | ppstructurev3
OCR_ENGINE=zhipu
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GLM_API_KEY` | ✅ | — | ZhipuAI GLM API key (also used for Rerank & OCR) |
| `GLM_BASE_URL` | ✅ | `https://open.bigmodel.cn/api/paas/v4` | GLM API endpoint |
| `DEEPSEEK_API_KEY` | ❌ | — | DeepSeek API key |
| `LLM_PROVIDER` | ❌ | `glm` | `glm` or `deepseek` |
| `LLM_MODEL` | ❌ | `glm-4` | Default model name |
| `LLM_TIERED_ROUTING` | ❌ | `false` | Enable fast/power model tiers |
| `LLM_FAST_MODEL` | ❌ | `deepseek-v4-flash` | Fast tier for intermediate steps |
| `LLM_POWER_MODEL` | ❌ | `deepseek-v4-pro` | Power tier for final answers |
| `WEB_API_KEY` | ❌ | — | Bearer token for API auth |
| `DESENSITIZE_ENABLED` | ❌ | `true` | PII masking for LLM calls |
| `OCR_ENGINE` | ❌ | `zhipu` | OCR engine selection |
| `SEARCH_DEFAULT_LIMIT` | ❌ | `10` | Default result count |
| `LOG_LEVEL` | ❌ | `INFO` | Logging level |
| `MAX_WORKERS` | ❌ | `4` | Thread pool size |

## Project Structure

```
src/
├── cli.py              # Click CLI (batch-convert / build-index / query / watch / stats)
├── api.py              # FastAPI server (21+ routes, SSE streaming, file upload)
├── mcp_server.py       # FastMCP server (4 tools, auto-index discovery)
├── agent/              # SearchAgent + 7 tools + LLMClient
│   ├── search_agent.py     # Agent loop (COMPILOT P0-P6, 8 rounds)
│   ├── llm_client.py       # LiteLLM wrapper + Tiered Routing
│   ├── analysis_agent.py   # Document analysis
│   └── tools/              # search, grep, read, rerank, summarize, bash, analyze
├── converter/          # Document → Markdown pipeline
│   ├── coordinator.py      # Auto-router + OCR fallback
│   ├── pdf.py, office.py, html.py, csv.py, text.py, image.py, msg.py, archive.py
│   └── ocr.py              # 4 engines
├── search/             # Search pipeline
│   ├── bm25_search.py      # BM25 (jieba + Bigram + title boost)
│   ├── hybrid.py           # BM25+Grep RRF fusion
│   ├── multi_index.py      # Multi-index search
│   ├── query_router.py     # Keyword routing (zero LLM)
│   └── reranker.py         # ZhipuAI cloud Rerank
├── storage/            # Persistence layer
│   ├── index.py            # Tantivy BM25 index (schema v2)
│   ├── convert_db.py       # SQLite (schema v2.1)
│   └── markdown_store.py   # Markdown storage
├── web/                # Web UI (zero-build vanilla HTML/CSS/JS)
│   ├── auth.py             # API key auth
│   ├── session_manager.py  # Session CRUD
│   ├── sse_events.py       # 11 SSE event types
│   ├── intent_classifier.py # Query intent routing
│   ├── upload_manager.py   # File upload pipeline
│   └── static/             # HTML, CSS, JS, i18n
├── stats/              # Usage + diagnostics + budget
│   ├── usage_tracker.py    # OCR/LLM/Rerank tracking
│   ├── budget_guard.py     # Budget enforcement
│   ├── search_logger.py    # Search logging
│   └── diagnostics.py      # 14-step timing
├── security/           # PII desensitization
│   ├── desensitizer.py     # Unified entry
│   └── maskers.py          # PII/Keyword/Regex maskers
├── watch/              # Directory monitoring
│   └── index_watcher.py    # Watchdog → incremental index
└── utils/              # Config / hash / tools
    ├── config.py           # Multi-provider LLM config
    ├── hash.py             # File/content hashing
    └── dir_diff.py         # Directory comparison
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
# Run tests
.venv\Scripts\python.exe -m pytest tests/ -q --tb=short

# Run tests with coverage
.venv\Scripts\python.exe -m pytest tests/ --cov

# Lint
.venv\Scripts\ruff check src/ tests/

# Format
.venv\Scripts\ruff format src/ tests/
```

## License

MIT License — see [LICENSE](LICENSE).
