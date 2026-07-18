# Stats 模块

## OVERVIEW
API 用量统计模块。统一追踪 OCR/LLM/Rerank token 用量，自动计算费用 (millicents)，支持预算控制与多格式报告导出。

## STRUCTURE
```
stats/
├── __init__.py        # 模块导出
├── usage_tracker.py   # UsageTracker — 统一 API 追踪 (OCR/LLM/Rerank)
├── budget_guard.py    # BudgetGuard — 月度/总预算监控与阻断
├── reporter.py        # StatsReporter — JSON/CSV/Markdown/HTML 导出
├── diagnostics.py     # DiagnosticsCollector — 14步分步计时 + LLM 调用详情持久化
├── search_logger.py   # SearchLogger — 异步搜索记录 (session_id + .md 训练格式 + SQLite)
├── memory.py          # AgentMemory (v0.21+) — 基于 search_logs.db 的历史问答召回
└── alerting.py        # AlertManager (v0.19+) — webhook 告警通知
```

## WHERE TO LOOK
| 任务 | 文件 | 关键 |
|------|------|------|
| LLM 用量记录 | usage_tracker.py | record_llm(model, input_tokens, output_tokens, cost, session_id) |
| Rerank 用量记录 | usage_tracker.py | record_rerank(model, query, num_docs, usage, session_id) |
| OCR 用量记录 | usage_tracker.py | record_ocr(model, input_tokens, output_tokens, cost) |
| 用量查询 | usage_tracker.py | get_summary(days), get_daily_usage(days), get_model_breakdown() |
| 价格配置 | usage_tracker.py | PricingConfig — 模型价格映射 (millicents per token) |
| 预算检查 | budget_guard.py | check_budget(budget_name) → BudgetStatus (ok/warning/blocked) |
| 预算设置 | budget_guard.py | set_budget(name, limit_cents, period="monthly") |
| 月度重置 | budget_guard.py | _reset_monthly_if_needed() — 自动检测月份变更 |
| 报告导出 | reporter.py | StatsReporter.export(format, output_path) |
| JSON 报告 | reporter.py | _export_json() — 完整用量数据 |
| CSV 报告 | reporter.py | _export_csv() — 每日用量 + 模型明细 |
| Markdown 报告 | reporter.py | _export_markdown() — 人类可读摘要 |
| HTML 报告 | reporter.py | _export_html() — 带样式的可视化报告 |
| 查询诊断 | diagnostics.py | DiagnosticsCollector — 14步分步计时, 持久化到 convert.db query_diagnostics 表 |
| 诊断步骤 | diagnostics.py | classify→query_analysis→decompose→expand→tool_loop→verify_draft→verify_recovery→confidence_calibration→best_of_k |
| LLM 调用日志 | diagnostics.py | llm_call_log 表 — call_type, latency_ms, tokens, retry_count, cache_hit |
| 搜索记录 | search_logger.py | SearchLogger — 异步 fire-and-forget, session_id srch_YYYYMMDD_HHMMSS_6hex |
| 训练数据 | search_logger.py | .md 文件 (4段格式: Instruction/Context/Reasoning/Response) + search_logs.db |
| 关闭日志 | search_logger.py | 三级: CLI --no-log / API log=false / NO_SEARCH_LOG=1 |
| 历史问答召回 | memory.py | AgentMemory.recall(query) → 精确命中直接返回(零延迟) / 模糊匹配注入 context |
| 学习策略 | memory.py | AgentMemory.learn() — 执行后自动记录搜索策略到 search_logs.tags |
| 用户反馈 | memory.py | AgentMemory.feedback() — 1-5 星评分，记录到 answer_feedback 表 |
| 告警通知 | alerting.py | AlertManager — webhook 通知错误/预算超限/索引健康 |
| 限速去重 | alerting.py | 同类告警 5 分钟内不重复发送 |

## CONVENTIONS
- **Millicents 计费**: 所有费用以 millicents (0.001 分) 为单位，避免浮点精度问题
- **UsageTracker 接入点**: LLMClient.chat_with_tools(), Reranker.rerank(), OCRService.process()
- **UsageTracker 可选**: 所有调用方 usage_tracker=None 时静默跳过，零开销
- **BudgetGuard 双级**: monthly (月度) + total (总计) 两种预算周期
- **BudgetGuard 状态**: ok → warning (80%) → blocked (100%)
- **BudgetGuard 自动重置**: 检测月份变更自动重置月度计数器
- **StatsReporter 四格式**: json (完整数据) / csv (表格分析) / md (可读摘要) / html (可视化)
- **CLI stats 命令组**: summary / daily / models / export / budget / realtime
- **convert_db 依赖**: UsageTracker 通过 ConvertDB 的 pricing/budget 表持久化

## ANTI-PATTERNS
- 不要用浮点数直接累加费用 — 用 millicents 整数运算
- 不要假设 UsageTracker 必须存在 — usage_tracker=None 是合法值
- 不要在 record 方法中做耗时操作 — 异步写入或批量写入
- 不要硬编码模型价格 — 通过 PricingConfig 从 pricing 表读取
- 不要跳过预算检查 — 超预算调用应该被阻断
- 不要混淆 millicents 和 cents — 1 cent = 1000 millicents
- 不要在报告导出中依赖外部库 — 纯标准库实现