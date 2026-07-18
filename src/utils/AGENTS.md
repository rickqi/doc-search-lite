# Utils 模块

## OVERVIEW
配置管理 + 工具函数。多 Provider LLM 配置 (GLM/DeepSeek + Tiered Routing)、密码字典 (加密文件解压)、目录对比迁移、文件监控、哈希工具。

## STRUCTURE
```
utils/
├── config.py          # 多 Provider 配置 + Tiered Routing (LLM_MODEL/LLM_FAST_MODEL/LLM_POWER_MODEL)
├── password_dict.py   # PasswordDictionary — 内置 45+ 密码 + 外部文件扩展
├── dir_diff.py        # 目录对比 — 内容哈希识别新增/变更/删除/移动
├── file_watcher.py    # FileWatcher — watchdog 目录监控 + 递归扫描 + symlink 去重
└── hash.py            # 内容哈希工具
```

## WHERE TO LOOK
| 任务 | 文件 | 关键 |
|------|------|------|
| LLM 配置 | config.py | config.litellm_model — 自动拼接前缀 (zai/glm-4 或 deepseek/deepseek-chat) |
| Tiered Routing | config.py | LLM_FAST_MODEL (中间步骤) / LLM_MODEL (默认) / LLM_POWER_MODEL |
| Provider 切换 | config.py | LLM_PROVIDER=glm 或 deepseek, Rerank/OCR 始终 GLM key |
| 密码字典 | password_dict.py | PasswordDictionary() — 内置默认 → 环境变量文件 → CLI 文件 (去重保序) |
| 目录对比 | dir_diff.py | DirectoryDiffer.compare() — SHA256 内容哈希, 识别 added/changed/deleted/moved |
| 文件监控 | file_watcher.py | FileWatcher._scan_directory_recursive() — path.resolve() 去重 symlink |
| .env 加载 | config.py | load_dotenv(override=False) — 不会覆盖已有环境变量 |

## CONVENTIONS
- **config.litellm_model**: 自动拼接 provider 前缀 — GLM → `zai/glm-4`, DeepSeek → `deepseek/deepseek-chat`
- **Rerank/OCR 固定 GLM**: Rerank API 和 OCR 始终用 GLM_API_KEY，不受 LLM_PROVIDER 影响
- **密码文件格式**: UTF-8, 每行一个密码, `#` 注释, 空行忽略
- **dir_diff 移动检测**: 同内容哈希 + 不同路径 = moved (非 deleted+added)
- **file_watcher symlink**: `path.resolve()` 去重, Linux 上符号链接解析到同一文件

## ANTI-PATTERNS
- **不要在 Router 路径运行时覆盖 api_base** — Router 初始化时已为每个 tier 配独立 api_base, 运行时覆盖导致 DeepSeek→GLM 401
- **不要用 os.environ.pop() 清除认证变量** — load_dotenv(override=False) 会重新填充, 改用 os.environ["KEY"] = ""
- **不要修改 Reranker 走 DeepSeek** — ZhipuAI Rerank 只认 GLM key
- **不要硬编码 provider 前缀** — 通过 config.litellm_model 构建
