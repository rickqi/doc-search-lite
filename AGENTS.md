# AGENTS.md — doc-search-lite

doc-search 开源核心，MIT 协议。PDF/DOCX/XLSX/PPTX/HTML → Markdown → Tantivy BM25 → Agent/LLM 搜索。
**无向量数据库，无本地模型推理**。企业版去除了 Pi TUI、Dify 集成、PDF 增强管线、QA 脚本等。

## 快速命令

```bash
.venv\Scripts\pytest tests/ -q --tb=short                    # 运行测试
.venv\Scripts\ruff check src/ tests/                         # Lint
.venv\Scripts\python -m src.cli batch-convert SRC --raw-root OUT  # 转换
.venv\Scripts\python -m src.cli build-index RAW_DIR          # 构建索引
.venv\Scripts\python -m src.cli query "问句" -i INDEX --agent  # Agent搜索
.venv\Scripts\python -m src.api                              # Web服务
.venv\Scripts\python -m src.mcp_server                       # MCP服务
```

## 入口

| 入口 | 命令 | 文件 |
|------|------|------|
| CLI | `doc-search-lite` (pyproject.toml scripts) | `src/cli.py` |
| API | `python -m src.api` | `src/api.py` — FastAPI 21+ routes |
| MCP | `python -m src.mcp_server` | `src/mcp_server.py` — 4 tools |
| Web | `python -m src.api` → `http://127.0.0.1:8000` | `src/web/static/` |

## 架构要点

- **无 packages 配置**: `pyproject.toml` 无 `[tool.setuptools.packages]`，用 `include = ["src*"]`。导入全用 `from src.xxx`
- **三个搜索结果类型**, 别混用: `SearchPreview` (bm25), `UnifiedSearchResult` (hybrid/multi), `SearchResult` (agent pipeline)
- **循环依赖警告**: `hybrid.py` 懒导入 `agent/tools/grep.py` — 不能在模块顶层 import
- **Rerank 只认 GLM key**: ZhipuAI Rerank API 和 OCR 始终用 `GLM_API_KEY`，不受 `LLM_PROVIDER` 影响

## CI (已配置)

`.github/workflows/ci.yml`: ruff check + pytest on push/PR. Python 3.10-3.12, ubuntu-latest only.

## 测试现状 (lite)

`tests/` 目录有 80 个 .py 文件（从企业版同步），但 **仅 `test_desensitization.py` 可独立运行**（31 tests）。
多数测试依赖企业版专有模块或需真实 Tantivy 索引。`fail_under=70` 覆盖率阈值不可达。

**conftest.py 提供的 fixture**:
- `mock_config` — `MagicMock(spec=Config)` 含默认值
- `_env_snapshot` — autouse, 自动恢复 `os.environ`

## 关键约束

| 约束 | 原因 |
|------|------|
| `os.environ["KEY"]=""` 非 `pop()` | `load_dotenv(override=False)` 会重填充 pop 掉的键 |
| 不修改 Reranker 走 DeepSeek | ZhipuAI Rerank API 只认 GLM key |
| 不在 Router 运行时覆盖 `api_base` | 会导致跨 provider 403 |
| 不放宽 `litellm>=1.90.0,<2.0.0` | 旧版 Router 已知问题 |
| 不在同一索引路径创建多个 writer | Tantivy LockBusy |

## 子模块 AGENTS.md

各模块有详细 AGENTS.md（从企业版同步），优先参考：
- `src/agent/AGENTS.md` — COMPILOT P0-P6, tool_loop, llm_client
- `src/converter/AGENTS.md` — 转换管线, OCR 引擎, pipeline_version
- `src/search/AGENTS.md` — BM25/Hybrid/Multi-index/Rerank
- `src/web/AGENTS.md` — API routes, SSE events, auth
- `src/storage/AGENTS.md` — Tantivy schema, ConvertDB schema v2.1
- `src/stats/AGENTS.md` — UsageTracker, BudgetGuard, Diagnostics

## 已知不存在的（别引用）

lite 已移除: `dify_retrieval.py`, `pi_bridge.py`, `tui.py`, `scripts/`, `opencode-skill/`, `processor/`
