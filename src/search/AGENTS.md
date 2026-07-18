# Search 模块

## OVERVIEW
搜索管线：BM25 全文检索 (jieba + Bigram 回退 + 标题加权) → Hybrid (BM25+Grep RRF融合, 可配置搜索策略) → Multi-index (QueryRouter 路由) → Rerank 重排序 → Benchmark 对比。

## STRUCTURE
```
search/
├── bm25_search.py      # BM25Searcher, SearchPreview, PaginatedResults, create_searcher() — title_boost 透传
├── unified.py          # SearchSource 枚举, UnifiedSearchResult/Results — 跨源统一模型
├── hybrid.py           # HybridSearcher — BM25+Grep 并行 RRF 融合 (K=60) + PROFILE_WEIGHTS
├── multi_index.py      # MultiIndexSearcher — 多索引扇出 + QueryRouter 路由
├── query_router.py     # QueryRouter — 关键词路由 (零 LLM 开销)
├── reranker.py         # ZhipuAIReranker — 云端 Rerank API (urllib, 含 UsageTracker)
├── benchmark.py        # BenchmarkRunner, QuerySpec, ModeResult, BenchmarkResult
├── report.py           # BenchmarkReporter — Markdown/HTML/JSON 三种格式
├── query_parser.py     # QueryParser — jieba 分词 + 短语/通配符/字段查询 + Bigram 回退
└── result_formatter.py # ResultFormatter — JSON/text/markdown 格式化
```

## WHERE TO LOOK
| 任务 | 文件 | 关键 |
|------|------|------|
| BM25 搜索 | bm25_search.py | BM25Searcher.search() → PaginatedResults, title_boost 参数 |
| 全文加载 | bm25_search.py | get_full_content() — 按需加载完整内容 |
| 混合搜索 | hybrid.py | RRF_K=60, PROFILE_WEIGHTS (legal/technical/faq/general) |
| 搜索策略 | hybrid.py | PROFILE_WEIGHTS: legal(bm25=1.0, grep=0.3), technical(1.0, 0.5), faq(0.6, 0.8), general(1.0, 0.5) |
| 多索引 | multi_index.py | ThreadPoolExecutor 并行, doc_id 命名空间: {idx}::{id} |
| 查询路由 | query_router.py | QueryRouter.route() — 关键词匹配索引元数据, 零 LLM 开销 |
| 标签路由 | query_router.py | route_by_tags() — 查询关键词匹配文档标签, --search-mode tag |
| Rerank | reranker.py | ZhipuAI /paas/v4/rerank, 始终用 GLM key, UsageTracker 集成 |
| 基准测试 | benchmark.py | load_queries(JSONL), run(modes), aggregate_by_mode() |
| 报告生成 | report.py | generate(result, fmt) → text/markdown/html/json |
| 查询解析 | query_parser.py | parse() — 短语"..."、字段field:value、通配符*?, Bigram 回退 |
| Bigram 回退 | query_parser.py | jieba OOV → 字符 bigram 生成, 补充提升中文专有名词召回 |
| 搜索高亮 | index.py (storage) | Tantivy Snippet + 查询词提取 |

## CONVENTIONS
- **PaginatedResults 构造**: 必须传 offset, limit, has_more — 签名变更曾导致测试失败
- **RRF 融合**: 只用排名位置，不需要分数归一化
- **Grep 评分**: 合成评分 log1p(match_count)/log1p(50)，用于 RRF 排名
- **Reranker**: 用 urllib 直接调用，不依赖额外 HTTP 库
- **Benchmark JSONL**: 每行 {"query": "...", "expected_files": [...], "category": "..."}
- **搜索模式**: auto(默认)/bm25/grep/hybrid/**tag**，CLI --search-mode 切换
- **中文分词**: jieba 预处理 → Tantivy 默认分词器 → Bigram 回退 (OOV 术语)
- **标题加权**: BM25Searcher 接受 title_boost 参数，Tantivy 查询 Boost 实现
- **QueryRouter 双路由**: route() 关键词匹配索引元数据; route_by_tags() 匹配文档标签 (tag 模式)
- **tag 模式**: 零 BM25 开销，用查询关键词直接匹配文档元数据中的 tags 字段
- **UsageTracker 可选**: reranker.py 的 usage_tracker=None 时完全不影响行为

## ANTI-PATTERNS
- 不要忘记 PaginatedResults 的 offset/limit/has_more 参数
- 不要修改 Reranker 走 DeepSeek — ZhipuAI Rerank API 只认 GLM key
- 不要对 get_full_content() 批量调用 — 只在用户选择时加载
- 不要跳过 min_score 过滤 — 低质量结果影响性能
- 不要在查询解析器返回空时使用原始查询 — 检查 parse() 返回值
- **循环依赖警告**: hybrid.py 懒导入 agent/tools/grep.py — 不要在模块顶层导入
- 三种搜索结果类型: SearchPreview (bm25), UnifiedSearchResult (unified), SearchResult (agent) — 不要混淆
- 不要假设 jieba 能处理所有中文术语 — OOV 用 Bigram 回退补充
- 不要在 QueryRouter 使用 LLM — 纯关键词路由，零成本