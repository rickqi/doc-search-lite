# AGENTS.md — doc-search-lite

本地文档智能搜索，**开源轻量版**。PDF/DOCX/XLSX/PPTX/HTML → Markdown → Tantivy BM25 索引 → Agent/LLM 搜索。**无向量数据库，无本地模型推理**。

从 [doc-search](https://github.com/rickqi/doc-search) 提取的核心版本，去除了企业功能和内部数据。

## 快速命令

```bash
.venv\Scripts\pytest tests/ -q --tb=short                    # 运行测试
.venv\Scripts\python -m src.cli batch-convert SRC --raw-root OUT  # 转换文档
.venv\Scripts\python -m src.cli build-index RAW_DIR               # 构建索引
.venv\Scripts\python -m src.cli query "问句" -i INDEX --agent     # Agent搜索
.venv\Scripts\python -m src.api                                    # Web服务
.venv\Scripts\python -m src.mcp_server                             # MCP服务
```

## 关键约定

**导入**: 全用 `from src.xxx`，包名是 `src`，`pyproject.toml` 无 `packages` 配置。

**环境变量 (.env)**: `GLM_API_KEY`（必需，也用于 Rerank/OCR）、`DEEPSEEK_API_KEY`（可选）、`LLM_PROVIDER=glm|deepseek`、`WEB_API_KEY`、`DESENSITIZE_ENABLED=true|false`。

**跨平台**: Win 主开发。Linux 注意 — `os.environ["KEY"]=""` 非 `pop()`（`load_dotenv` 重填充）。

**License**: MIT。主项目 doc-search 使用 PolyForm Strict。

## 模块结构

```
src/
├── cli.py              # Click 入口
├── api.py              # FastAPI
├── mcp_server.py       # FastMCP
├── security/           # PII 脱敏
├── converter/          # 文档转换 → Markdown
├── agent/              # SearchAgent (tool_loop + COMPILOT P0-P6)
│   └── tools/          # search/read/grep/rerank/bash/summarize
├── search/             # BM25/Hybrid/Multi-index/ABTestRunner
├── web/                # vanilla HTML/CSS/JS + SSE
├── stats/              # UsageTracker/BudgetGuard/Diagnostics/SearchLogger/AgentMemory
├── storage/            # Tantivy Index + SQLite ConvertDB
├── utils/              # Config/FileWatcher/Hash
└── watch/              # watchdog 增量索引
```

## 与 doc-search 的区别

| 特性 | doc-search (内部) | doc-search-lite (开源) |
|------|:-:|:-:|
| License | PolyForm Strict | **MIT** |
| PDF 增强管线 (LA-3B) | ✅ | ❌ |
| Pi TUI | ✅ (已弃用) | ❌ |
| Dify 集成 | ✅ | ❌ |
| COS 备份 | ✅ | ❌ |
| OpenCode Skill | ✅ | ❌ |
| 内部测试数据 | ✅ | ❌ |
| 设计文档 | ✅ | ❌ |

## 同步策略

- 核心代码从 doc-search 同步到 lite: `scripts/sync-lite.py`
- 社区贡献直接在 lite 中开发
- 企业功能仅在 doc-search 中开发

## 不要做的事

- 不修改 Reranker 走 DeepSeek — 只认 GLM key
- 不在同一索引路径创建多个 writer（LockBusy）
- 不放宽 `litellm>=1.90.0,<2.0.0` — 旧版 Router 已知问题
- 不在 Router 运行时覆盖 `api_base` — 会导致跨 provider 403
- 不在 dify record 中返回 null metadata — Dify 要求 `{}`
- 不在 conftest.py 中用 `os.environ.pop()` — 改用 `["KEY"]=""`
