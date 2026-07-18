# Web 模块

## OVERVIEW
Web 浏览器界面。FastAPI + SSE 流式事件 + vanilla HTML/CSS/JS 前端，21 条 API 路由，双模式查询路由（文档搜索/合规审查），SQLite 会话持久化支持跨进程。

## STRUCTURE
```
web/
├── auth.py             # API Key 认证中间件 (WEB_API_KEY)
├── session_manager.py  # 会话管理 (SQLite 持久化, 20上限, 30min超时)
├── session_store.py    # SessionStore (跨进程 SQLite, WAL模式)
├── sse_events.py       # SSE 事件协议 (11 种事件类型)
├── review_prompts.py   # 合规审查检测 + Prompt 增强
├── upload_manager.py   # 文件上传 Pipeline (上传→转换→索引)
└── static/             # 前端 (零构建 vanilla HTML/CSS/JS)
    ├── index.html      # 双标签布局 (会话+数据库)
    ├── app.js          # SSE 客户端 + 双模式路由 + 搜索结果渲染
    ├── style.css       # 暗/亮主题 + 移动端响应式
    └── i18n.js         # 国际化支持 (zh-CN/en, ?lang=en)
```

## WHERE TO LOOK
| 任务 | 文件 | 关键 |
|------|------|------|
| 会话 CRUD | session_manager.py | SessionManager.create/get/delete/list |
| SSE 事件流 | sse_events.py | AgentEventType (11种) + sse_encode() |
| 合规检测 | review_prompts.py | detect_review_mode() 8特征词≥3触发 |
| Prompt 增强 | review_prompts.py | REVIEW_ENHANCEMENT 5类规则×8条指引 |
| 认证 | auth.py | WebAuthMiddleware (Bearer/X-API-Key) |
| 上传 | upload_manager.py | UploadManager (SSE进度流) |
| 前端渲染 | app.js | finalizeAnswer() — 搜索命中详情卡片 |
| 导出 | app.js | exportResult() — Markdown 下载 |
| 技能选择 | index.html | #skill-select — 内置6种 + 外部 __load__ |
| (lite 已移除) | — | Dify 外部知识库 API 不在 lite 中 |

## CONVENTIONS
- SSE: 50字符分片, 20ms间隔, 30s心跳保活
- 会话存储: sessions.db (SQLite WAL), 跨进程共享
- 前端: Chart.js CDN, 零构建
- 认证: WEB_API_KEY 设置后受保护端点需 Bearer/X-API-Key
- answer_complete 事件含 search_hits 结构化字段
- (lite 已移除 Dify 集成)

## ANTI-PATTERNS
- 不要在前端引入构建工具 — 零编译原则
- 不要用 WebSocket — SSE 足够（单向推送）
- 不要在前端判断查询模式 — 后端 detect_review_mode() 自动路由
