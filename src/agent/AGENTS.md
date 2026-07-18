# Agent 模块

## OVERVIEW
智能搜索 Agent 系统。SearchAgent 通过 tool_loop (8轮, 动态置信度) 自主调用工具搜索文档并回答问题，支持 LLM 多 Provider (GLM/DeepSeek)，查询扩展 + UsageTracker 统计。

**Agent Loop 优化 (v0.14-v0.15)**: 基于 COMPILOT 论文 6 项改进 — ReAct 推理 (P0), Draft 验证闭环 (P1), 工具反馈信号 (P2), 收敛推促 (P3), Query 前置分析 (P4), Best-of-K (P5), Confidence 校准 (P6)。

## STRUCTURE
```
agent/
├── base.py              # Tool ABC, Agent ABC, AgentResponse, ToolResult, ToolCache (TTL 300s)
├── llm_client.py        # LLMClient — litellm 封装, chat_with_tools, _ToolCallCache (60s去重)
├── search_agent.py      # SearchAgent — tool_loop/pipeline, 查询扩展, 动态置信度, UsageTracker session
├── analysis_agent.py    # AnalysisAgent — compare/extract/summarize
├── query_decomposer.py  # QueryDecomposer — complex 查询分解为独立子查询
├── sufficient_context.py # SufficientContextChecker — 充足性判断 (三重检查)
├── skill_loader.py      # 外部 SKILL.md 发现和加载
└── tools/
    ├── search.py        # SearchTool (BM25 封装, TTL 缓存)
    ├── grep.py          # GrepTool (Python re, TTL 缓存)
    ├── read.py          # ReadTool (MarkdownStore 封装, TOC 注入 on first read)
    ├── bash.py          # BashTool (模拟只读 shell, 中文错误消息)
    ├── rerank.py        # RerankTool (ZhipuAI Rerank 封装)
    ├── summarize.py     # SummarizeTool (LLM 文档摘要, 节省 token)
    └── analyze.py       # AnalyzeTool (LLMClientProtocol, 仅 AnalysisAgent 使用)
```

## WHERE TO LOOK
| 任务 | 文件 | 关键 |
|------|------|------|
| Tool 接口 | base.py | Tool: name, description, execute(**kw), to_openai_tool() |
| 工具缓存 | base.py | ToolCache (TTL 300s, LRU 128, MD5 key) |
| LLM 调用 | llm_client.py | chat(), chat_with_tools(), _compress_conversation() |
| 工具去重 | llm_client.py | _ToolCallCache (TTL 60s, max 50, 每次 chat_with_tools 新建) |
| Agent 主流程 | search_agent.py | run() → _run_tool_loop() (**8轮**, 非100轮) 或 _run_pipeline() |
| 查询扩展 | search_agent.py | _expand_query() — LLM 生成 2-3 同义查询注入 system message |
| 动态置信度 | search_agent.py | _calculate_tool_loop_confidence() — 启发式 60% + LLM 自评 40% (complex only) |
| Best-of-K (P5) | search_agent.py | run() L335-361 — complex + confidence < 0.5 → K=3 取最优 |
| Confidence 校准 (P6) | search_agent.py | _llm_self_assess_confidence() L1494 — fast-tier LLM 自评 (仅 complex) |
| Draft 验证闭环 (P1) | search_agent.py | _verify_draft_grounding() L1364 → 失败时自动搜索→重生成 L836-901 |
| ReAct 推理 (P0) | search_agent.py | SYSTEM_PROMPT L36-103 — Thought→Action 结构化推理 |
| 工具反馈信号 (P2) | tools/search.py, grep.py, read.py | 零命中 hint, 重复读取检测, 多词 OR 建议 |
| 收敛推促 (P3) | search_agent.py | _consecutive_searches ≥ 1 且无 read → 首次推促, 二次硬停 |
| 强制读取护栏 | search_agent.py | L670-727 — tool_loop 无 read → 强制读 top 3 + 重生成 |
| 答案质量护栏 | search_agent.py | L784-801 — 9 种低质量模式检测 + 重生成 |
| 复杂度分级 | search_agent.py | _classify_query_complexity() — 4 级: simple(2轮)/light(4轮)/medium(8轮)/complex(8轮+分解+验证+BOK) |
| Skill 注入 | search_agent.py | SKILL_PROMPTS (6种) + _build_system_prompt() |
| 工具注册 | search_agent.py | create_search_agent() — Search/Read/Grep/Bash/Rerank (UsageTracker 透传) |
| Token 预算 | search_agent.py | _max_session_tokens=50000, 每次 run() 重置 |
| 上下文压缩 | llm_client.py | _compress_conversation(level0-level5, 默认level3) |
| 用量追踪 | llm_client.py | usage_tracker 参数 → record_llm() 每次 chat |
| 查询分解 | query_decomposer.py | QueryDecomposer.decompose() — LLM 分解 complex 查询为子查询 |
| 充足性判断 | sufficient_context.py | SufficientContextChecker.check() — 三重检查 (覆盖+细节+缺失) |
| Tiered Routing | llm_client.py | _resolve_model(tier) → fast/default/power, auth 失败自动冷却 5min |
| TOC 注入 | tools/read.py | _format_toc() + _load_headings(), 首次读取 (start_line==0) 时注入 |
| tool_calls metadata | search_agent.py | on_tool_call 回调保存完整 metadata (lines_read, start_line, execution_time) |

## CONVENTIONS
- **Tool 协议**: 实现 Tool ABC 的 name, description, execute(**kwargs), to_openai_tool()
- **ToolResult 工厂**: 用 ToolResult.ok()/ToolResult.fail()，不要直接构造
- **Agent 响应**: AgentResponse(success, answer, sources, tool_calls, reasoning, tokens_used)
- **多 Provider**: config.litellm_model 自动拼接前缀 (zai/glm-4 或 deepseek/deepseek-chat)
- **Reranker 条件注册**: `reranker.available=True` (即 GLM_API_KEY 已配置) 时注册 RerankTool, LLM 自行决定是否调用。`RERANKER_TYPE=local` 切换本地 bge-reranker (默认 zhipu 云端)
- **Skill 系统**: --skill (内置6种) + --load-skill (外部SKILL.md)，两者可叠加
- **工具结果**: 返回 JSON 字符串给 LLM
- **复杂度 4 级** (v0.14+): simple(2轮,跳过扩展) / light(4轮,跳过扩展) / medium(8轮) / complex(8轮+分解+验证+Best-of-K)
- **self._complexity 依赖**: `_run_tool_loop()` 设置 `self._complexity`, Best-of-K 和 P6 Confidence 读取此属性 — 重构时不要移除
- **Best-of-K 常量**: `BEST_OF_K_THRESHOLD = 0.5`, `BEST_OF_K_RUNS = 3` — 不要改回 0.7 (会导致 3x rerun)
- **_no_log 属性**: MCP server 外部设置 `agent._no_log = True` 禁用 SearchLogger — 不是构造参数
- **内容截断**: tool_calls 条目 [:2000], 重生成文档 [:3000], 验证文档单条 [:2000] 总计 [:6000]
- **LLMClientProtocol**: AnalyzeTool 用 Protocol 接口 (generate/count_tokens)，不是 LLMClient 类
- **AnalyzeTool 未注册**: create_search_agent() 不注册 AnalyzeTool，它是 AnalysisAgent 专用
- **RerankTool 双类型**: documents 参数接受 JSON 字符串或 Python list
- **UsageTracker 可选**: usage_tracker=None 时完全不影响行为
- **懒导入**: ZhipuAIReranker/GrepTool/BashTool/RerankTool 在 create_search_agent() 内部导入
- **TYPE_CHECKING 守卫**: ZhipuAIReranker 用 TYPE_CHECKING 避免运行时循环导入
- **Multi-index**: create_search_agent 支持逗号分隔 index_path → 自动使用 MultiIndexSearcher
- **ReadTool**: raw_dirs 参数支持多个 raw 目录，source_path 未命中时依次尝试
- **ReadTool TOC**: 首次读取 (start_line==0) 自动注入 TOC 块（来自 .md.json headings 字段）
- **交替工作流** (v0.15.1+): SYSTEM_PROMPT 强制 search→read 交替, 禁止连续搜索

## 护栏链 (Guardrail Chain)
按执行顺序的安全网:
1. **收敛早停** — 连续搜索 top-3 重复 → 注入收敛消息
2. **收敛推促** (P3) — 连续搜索无 read → 首次推促 (非硬停), 二次硬停
3. **强制读取** — tool_loop 无 read → 强制读 top 3 + 重生成
4. **答案质量检测** — 9 种低质量模式 → 重生成
5. **Draft 验证** (P1) — complex 查询草稿验证 → 失败时自动搜索→重生成 (fail-open)
6. **Best-of-K** (P5) — complex + 低置信度 → K=3 取最优 (fail-safe, except: pass)

## ANTI-PATTERNS
- 不要直接调用工具 — 使用 execute_tool(name, **kwargs)
- 不要跳过 ToolResult 包装 — 始终返回 ToolResult.ok() 或 .fail()
- 不要修改 Reranker 走 DeepSeek — ZhipuAI Rerank API 只认 GLM key
- 不要硬编码 zai/ 前缀 — 通过 config.litellm_model 构建
- 不要忽略 sources 追踪 — AgentResponse 必须包含 sources
- 不要混淆三种搜索结果类型: SearchPreview (bm25), UnifiedSearchResult (hybrid/multi), agent.SearchResult (pipeline)
- 不要在同一个索引路径创建多个 TantivyIndexManager writer — 会锁冲突
- 不要用 zhipuai SDK — OCR 用 zai-sdk
- 不要假设 tool_loop 默认 100 轮 — 已改为 MAX_TOOL_ITERATIONS = 8
- 不要假设 max_tokens=800 — 已改为 2000
- **不要在 Router 路径运行时覆盖 api_base**: Router model_list 初始化时已为每个 tier 配独立 api_base，运行时覆盖会导致跨 provider 请求 (DeepSeek→GLM → 401)
- **不要只提取 execution_time**: on_tool_call 应保存完整 result.metadata (lines_read, start_line 等)
- **不要移除 `self._complexity = complexity` 赋值** — Best-of-K 和 P6 Confidence 静默依赖此属性
- **不要改 BEST_OF_K_THRESHOLD 回 0.7** — 0.7 导致 3x rerun，已降为 0.5
- **不要移除 Best-of-K 的 try/except: pass** — 必须 fail-safe，不能因 BOK 失败阻断查询
- **不要对非 complex 查询启用 P6 LLM 自评** — 增加 1s 延迟但收益微乎
- **不要改 Draft 验证的 fail-open 行为** — 错误时 sufficient=True (不阻断答案)
- **不要在 GrepTool 构造时传 `snippet_length`** — 该参数不存在，用 `max_results`
