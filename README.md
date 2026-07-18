# doc-search-lite

Local document intelligent search — **lightweight open-source edition**.

Convert PDF/DOCX/XLSX/PPTX/HTML → Markdown → BM25 index → LLM-powered search.

This is the open-source core of [doc-search](https://github.com/rickqi/doc-search),
stripped of enterprise features and internal data.

## Quick Start

```bash
git clone https://github.com/rickqi/doc-search-lite.git
cd doc-search-lite
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# Convert documents
python -m src.cli batch-convert ./docs --raw-root ./raw

# Build index
python -m src.cli build-index ./raw

# Search
python -m src.cli query "关键词" -i ./raw/index --agent
```

## License

MIT
