# Storage 模块

## OVERVIEW
存储和索引管理。Tantivy BM25 全文索引 (SCHEMA_VERSION="2", 标题加权 + 搜索高亮)，SQLite 转换状态持久化 (ConvertDB Schema v2.1, 含 pricing/budget 表 + query diagnostics)，Markdown 文件镜像存储。

## STRUCTURE
```
storage/
├── base.py           # DocumentRecord, SearchHit, SearchResult, Storage ABC
├── index.py          # TantivyIndexManager — SCHEMA_VERSION="2", 标题加权, 搜索高亮
├── convert_db.py     # ConvertDB — SQLite 多表 (Schema v2.1: +pricing +budget +query_diagnostics +llm_call_log)
├── markdown_store.py # MarkdownStore — 镜像存储, doc_id 索引, 图片管理
├── raw_store.py      # RawStore — 独立 raw 目录, 源目录无关
└── metadata.py       # MetadataManager — JSON index.json, 过滤查询
```

## WHERE TO LOOK
| 任务 | 文件 | 关键 |
|------|------|------|
| 索引增删改查 | index.py | add_document(), update_document(), delete_document(), search() |
| 索引 Schema | index.py | doc_id(raw), filename, title, content, keywords, source_path |
| 标题加权 | index.py | _build_query() 中 title 字段 Boost, title_boost 参数 |
| 搜索高亮 | index.py | search() 返回带高亮片段的结果, Snippet + 查询词提取 |
| 索引提交 | index.py | commit() + config_reader(reload_policy="commit") |
| 转换状态 | convert_db.py | upsert_file(), get_files_by_status(), create_batch() |
| pipeline_version | convert_db.py | PIPELINE_VERSION="3", _ensure_pipeline_version_column() 幂等迁移 |
| Schema v2.1 | convert_db.py | pricing 表 + budget 表 + query_diagnostics 表 + llm_call_log 表 |
| Token 统计 | convert_db.py | add_token_usage(), get_token_summary() — 含 cost/session_id/source_dir |
| Markdown 存储 | markdown_store.py | save(), load(), list_documents() |
| Doc ID 查找 | markdown_store.py | _build_doc_id_index() — 遇到非 dict JSON 跳过 |
| Raw 目录管理 | raw_store.py | save() — 映射 source_root/file → raw_root/source_name/file |

## CONVENTIONS
- **Schema 版本**: SCHEMA_VERSION="2" (Tantivy), **"2.1"** (ConvertDB，从 "1.1" → "2.0" → "2.1" 升级)
- **pipeline_version**: `"3"` (ConvertDB)，追踪转换管线步骤，用于 catalog repair 判断是否需重处理
- **doc_id 字段**: tokenizer_name="raw" 精确匹配，不参与分词
- **中文分词**: jieba 预处理后空格分隔，不注册自定义 Tantivy 分词器
- **readonly 模式**: CLI 搜索时不创建 writer，避免 LockBusy
- **标题加权**: index.py search() 接受 title_boost 参数 (默认 1.0)，通过 Tantivy Boost 实现
- **搜索高亮**: search() 结果包含高亮片段，使用 Tantivy Snippet API + 查询词提取
- **Doc ID**: SHA256(relative_path.as_posix())[:16]，使用正斜杠保证跨平台
- **文件冲突**: 自动加数字后缀 file_1.md, file_2.md
- **SQLite**: WAL 模式，每源目录独立 convert.db
- **Context manager**: ConvertDB 和 TantivyIndexManager 都支持 with 语句
- **路径差异**: RawStore 保留原扩展名 (file.xlsx.md)，MarkdownStore 替换扩展名 (file.md)
- **元数据**: .md 文件 → .md.json 伴随文件（注意是追加 .json 不是替换扩展名）
- **文件状态生命周期**: pending → converting → success / failed / skipped
- **启动恢复**: mark_interrupted_batches() 将 running 批次重置为 interrupted
- **pricing 表**: 模型价格映射 (input/output per million tokens, millicents)
- **budget 表**: 预算名称、限制、周期 (monthly/total)、已用金额

## ANTI-PATTERNS
- 不要在同一索引路径创建多个 IndexWriter — LockBusy 错误
- 不要修改文档后忘记 commit() — 否则搜索不到
- 不要在搜索前不 reload — config_reader(reload_policy="commit") 自动处理
- 不要忽略 ByteStringObject — PDF 元数据需 _sanitize_for_json()
- 不要假设元数据 JSON 一定是 dict — _build_doc_id_index() 会跳过非 dict
- 不要假设 ConvertDB Schema 是 "2.0" — 已升级到 "2.1" (新增 query_diagnostics/llm_call_log 表)
- **pipeline_version 列**: _ensure_pipeline_version_column() 独立幂等方法，不依赖 schema 迁移路径
- 不要忘记标题加权参数 — BM25Searcher 构造时传入 title_boost
- 不要跳过高亮逻辑 — 搜索结果需要关键词标记