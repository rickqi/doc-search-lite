# Watch 模块

## OVERVIEW
目录监控自动重索引。watchdog 监听 raw 目录 → .md 文件变更自动增量更新 Tantivy 索引。支持防抖、增量更新、日志文件输出。

## STRUCTURE
```
watch/
├── __init__.py        # 导出 IndexWatcher, start_watching
└── index_watcher.py   # watchdog 事件处理 + Tantivy 增量更新
```

## WHERE TO LOOK
| 任务 | 文件 | 关键 |
|------|------|------|
| 启动监控 | cli.py (watch 命令) | IndexWatcher.start(blocking=True) |
| 事件处理 | index_watcher.py | _MdFileHandler — on_created/modified/deleted/moved |
| 索引更新 | index_watcher.py | _add_to_index() / _remove_from_index() |
| 防抖 | index_watcher.py | _pending dict + debounce_seconds 参数 |
| CLI 入口 | cli.py | @cli.command("watch") + --debounce/--no-jieba/--log-file |

## CONVENTIONS
- doc_id = SHA256(rel_path.as_posix())[:16] — 与 build-index 一致
- 内容采样: _sample_content() — 50000字符, 5MB阈值
- .md 元数据: 读取 .md.json 获取 tags 用于 keywords 字段
- commit() 每次文件变更后立即提交

## ANTI-PATTERNS
- 不要在监控前不构建初始索引 — watch 只处理增量
- 不要在同一 raw 目录创建多个 watcher — LockBusy
- 不要调用 rebuild() — 增量更新不要重建整个索引
