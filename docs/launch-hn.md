# HN Launch Post

## Title

Show HN: doc-search-lite – Local document RAG without vector databases

## Body

I built a local document search system that converts PDFs, DOCXs, spreadsheets, and images into a BM25 index and lets an LLM agent search and answer questions from your own knowledge base.

**No vector database. No documents leave your machine. No $200/month OpenAI bills.**

### How it works

```
Your files → Markdown → Tantivy BM25 index → LLM agent (search/read/answer)
```

3 commands from zero to answers:

```bash
pip install -e ".[dev]"
python -m src.cli batch-convert ./docs --raw-root ./raw
python -m src.cli query "What's our remote work policy?" -i ./raw/index --agent
```

### Why not just throw everything into a vector DB?

For insurance policies, regulatory docs, and legal contracts, keyword precision matters more than semantic similarity. BM25 + jieba (Chinese tokenization) with Bigram fallback catches exact matches that embeddings often miss. Hybrid search (BM25 + Grep with RRF fusion) gives you both.

### What it does

- **11 formats**: PDF, DOCX, XLSX, PPTX, HTML, CSV, TXT, images (OCR), Outlook MSG, ZIP/7z/RAR
- **5 search modes**: BM25, Grep, Hybrid (RRF), Tag, Agentic RAG
- **3 interfaces**: CLI, Web UI (FastAPI + SSE), MCP server (OpenCode/Claude)
- **PII desensitization**: Phone/ID/bank card masking before LLM calls
- **Agent loop**: COMPILOT-inspired optimizations (ReAct, draft verification, Best-of-K, confidence calibration)
- **Cost tracking**: Usage metering in millicents, budget guard, 14-step diagnostics
- **Dual LLM**: GLM or DeepSeek, with tiered routing (cheap model for intermediate steps)

### Tech stack

Tantivy (Rust BM25) + LiteLLM + FastAPI + vanilla HTML/CSS/JS + SQLite

### Links

GitHub: https://github.com/rickqi/doc-search-lite
MIT licensed.
