# doc-search-lite

**Local document intelligent search** — lightweight open-source edition.

Convert PDF/DOCX/XLSX/PPTX/HTML → Markdown → BM25 index → LLM-powered search.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![CI](https://github.com/rickqi/doc-search-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/rickqi/doc-search-lite/actions/workflows/ci.yml)

---

## Why doc-search-lite?

This is the open-source core of [doc-search](https://github.com/rickqi/doc-search), stripped of enterprise features and internal data.

| Feature | doc-search (enterprise) | doc-search-lite (OSS) |
|---------|:----------------------:|:---------------------:|
| BM25 + Agent RAG search | ✅ | ✅ |
| Multi-format document conversion | ✅ | ✅ |
| Web UI + API + MCP | ✅ | ✅ |
| PII desensitization | ✅ | ✅ |
| PDF enhancement (LA-3B) | ✅ | ❌ |
| Dify integration | ✅ | ❌ |
| Pi TUI | ✅ (deprecated) | ❌ |
| License | PolyForm Strict | **MIT** |

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
```

## Architecture

```
User Query → BM25/Hybrid Search → Agent (8-round tool_loop) → LLM → Answer
                ↕                          ↕
         Tantivy Index (Rust)      PII Desensitization Layer
                ↕
         Document Converter
         (PDF/DOCX/XLSX/PPTX/HTML → Markdown)
```

## Commands

| Command | Description |
|---------|-------------|
| `batch-convert` | Convert documents to Markdown |
| `build-index` | Build Tantivy search index |
| `query` | Search (BM25 / Hybrid / Agent) |
| `ab-test` | A/B compare search configurations |
| `watch` | Auto-index on file changes |
| `python -m src.api` | Launch Web UI |
| `python -m src.mcp_server` | Start MCP server |

## Supported Formats

PDF, DOCX, XLSX, PPTX, HTML, CSV, TXT, Markdown, Images (OCR), Outlook MSG, ZIP/7z/RAR archives.

## License

MIT License — see [LICENSE](LICENSE).
