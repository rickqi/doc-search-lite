# Converter 模块

## OVERVIEW
文档转 Markdown。ConverterCoordinator 按扩展名自动路由到对应转换器，含 v2 转换管线 (表格对齐→OCR后处理→标签提取)，扫描 PDF 自动 OCR 回退，`pipeline_version` 追踪回溯修复。UsageTracker 集成。

## STRUCTURE
```
converter/
├── base.py            # Converter ABC, ConvertResult, UnsupportedFormatError
├── coordinator.py     # ConverterCoordinator — 自动路由, OCR 回退, 管线集成, UsageTracker
├── pdf.py             # PDFConverter — extract_images 默认 False, 按需渲染+OCR, 表格对齐
├── office.py          # OfficeConverter — MarkItDown (docx/pptx/xlsx)
├── html.py            # HTMLConverter — 含表格对齐修复 + HTML 合并单元格回退
├── csv.py             # CSVConverter — pandas, 自动编码检测
├── text.py            # TextConverter — 透传 + 编码检测
├── image.py           # ImageConverter — OCR (四引擎: zhipu/paddleocr/paddleocr-http/ppstructurev3) + UsageTracker
├── ocr.py             # OCRService (zai-sdk), PaddleOCRService (本地), PaddleOCRHTTPService (远程 GPU, 降级链 VL→Structure→OCR)
├── ocr_postprocess.py # OCR 后处理 4 步管道 (去噪→合并→标题→表格)
├── table_fix.py       # 表格对齐修复 + HTML 合并单元格回退 (解析 _tbl XML)
├── tag_extractor.py   # 零 LLM 关键词标签提取器 — 保险/监管领域模式匹配
├── headings.py        # 零 LLM H1-H6 标题提取 — 纯正则, ~0.5ms/50KB, MAX_HEADINGS=30
├── msg.py             # MsgConverter — olefile (Outlook .msg OLE2 流)
└── archive.py         # ArchiveConverter — ZIP/7z/RAR/tar 解压递归转换
```

## WHERE TO LOOK
| 任务 | 文件 | 关键 |
|------|------|------|
| 自动选择转换器 | coordinator.py | get_converter(ext) — SUPPORTED_EXTENSIONS 路由 |
| 扫描 PDF 检测 | coordinator.py | _is_scanned_pdf() — 文本密度 < 50 字符/页 |
| OCR 回退 | coordinator.py | _convert_scanned_pdf() — 调用 OCRService + OCR 后处理 |
| PDF 按需渲染 | coordinator.py | _render_pdf_pages() — OCR 时渲染 → `_ocr_temp/` → 用完自动清理 |
| 标签提取集成 | coordinator.py | TagExtractor.extract() → 元数据 JSON tags 字段 |
| UsageTracker 接入 | coordinator.py | usage_tracker 参数 → record_ocr() / record_llm() |
| 表格对齐修复 | table_fix.py | fix_table_alignment() — 补齐列数 |
| 合并单元格回退 | table_fix.py | fix_merged_tables_with_html() — 解析 _tbl XML, rowspan/colspan → HTML `<table>` |
| OCR 后处理 | ocr_postprocess.py | ocr_postprocess() — 4 步管道: 去噪→合并→标题→表格 |
| 标签提取 | tag_extractor.py | TagExtractor.extract() — 零 LLM, 保险/监管关键词匹配 |
| 标题提取 | headings.py | extract_headings() — 纯正则 H1-H6, 跳过代码块, MAX_HEADINGS=30 |
| Headings 存储位置 | coordinator.py | result.metadata["headings"] → .md.json 文件 |
| 已有知识库回填 | (CLI) | catalog backfill-headings — 只更新 .md.json，不重转换 |
| OCR 引擎选择 | ocr.py | 四引擎: zhipu ✅ / paddleocr ✅ / paddleocr-http ✅ / ppstructurev3 ⚠️ |
| OCR 双模式 | ocr.py | OCRMode.LAYOUT_PARSING (zai-sdk) / VISION_CHAT (litellm, glm-5-turbo) |
| OCR HTTP 远程 | ocr.py | PaddleOCRHTTPService — recognize_vl() / recognize_structure() / recognize() 降级链 |
| OCR VLM 解析 | ocr.py | recognize_vl() → POST /vl — VLM 最优质量, markdown + 图片检测 |
| OCR 结构解析 | ocr.py | recognize_structure() → /structure/download?format=md — 布局+表格+公式+印章 |
| PDF 表格修复 | pdf.py | 转换后调 fix_table_alignment() |
| HTML 表格修复 | html.py | 转换后调 fix_table_alignment() + fix_merged_tables_with_html() |
| 压缩文件解压 | archive.py | 安全限制: 路径穿越检查 + symlink 拒绝 + 10K文件上限 + 512MB |
| MSG 邮件解析 | msg.py | OLE2 流: __substg1.0_0037001F(Subject), __substg1.0_1000001F(Body) |
| OCR 用量追踪 | image.py | usage_tracker 参数 → record_ocr() 每次 OCR 调用 |

## CONVENTIONS
- 所有转换器返回 ConvertResult(success, markdown, images, metadata, errors)
- 转换后文件名保留原扩展名: file.docx.md
- pdfplumber/pypdf/PIL/py7zr/rarfile 懒加载 — 不在模块顶层 import
- OCR SDK: zai-sdk (ZhipuAiClient)，不是 zhipuai
- OCR 视觉模型: 默认 `glm-5-turbo` (OCRServiceConfig.vision_model)
- PDF 密码: 通过 options['password'] 传入
- 压缩文件: 解压到临时目录 → 递归转换支持格式 → 汇总 Markdown
- CSV/Text 编码检测: utf-8 → utf-8-sig → gbk → gb2312 → latin-1
- HTML-in-PDF 检测: .pdf 但以 `<!DOCTYPE`/`<html` 开头 → 路由到 HTMLConverter
- json.dumps 始终传 ensure_ascii=False — 保留中文字符
- 压缩文件密码保护: 检测到密码保护时优雅跳过，不报错
- **UsageTracker 可选**: coordinator.py 和 image.py 的 usage_tracker=None 时完全不影响行为
- **错误处理**: Result-based 模式 — ConvertResult.errors 收集错误，不抛异常
- **转换管线 v2**: 表格对齐 → OCR 后处理 → 标签提取 → 标题提取(headings)，按 pipeline_version 追踪
- **PDF 图片**: extract_images=False (默认)，OCR 时按需 `_render_pdf_pages()` → `_ocr_temp/` 自动清理
- **表格对齐**: grid map 先处理 vMerge=continue，再跳过已占位；continuation 只存 `{"skip": True}`

## ANTI-PATTERNS
- 不要在模块顶层 import pdfplumber/pypdf/py7zr/rarfile — 用懒加载 _get_X() 模式
- 不要用 zhipuai 包 — OCR 用 zai-sdk
- `.doc` 格式不保证 — MarkItDown 列为支持但质量不稳定，需先转 .docx
- 不要忽略 ByteStringObject — 用 _sanitize_metadata_value() 处理 PDF 元数据
- 不要在 ArchiveConverter 跳过安全检查 — 路径穿越和 symlink 必须拦截
- 不要跳过 ensure_ascii=False — json.dumps 默认会破坏中文
- 不要假设文件扩展名正确 — coordinator 有 HTML-in-PDF 误命名检测
- 不要假设 UsageTracker 始终存在 — usage_tracker 参数可选
- 不要在 `table_fix._build_grid_map` 里用 `__W_NS` — name mangling 会破坏模块级常量 `_W_NS`
- 不要在 PDF 转换中默认 extract_images=True — 按需渲染，用完自动清理
- 不要用 `fix_merged_tables_with_html` 做 catalog repair — 需要源 .docx，repair 只有 .md
- **paddleocr-http 不要并行**: `--parallel > 1` 导致 WSL GPU 服务崩溃 (Connection refused)
- **structure 不要默认 use_chart=true**: chart 分析增加 3-10x 延迟 (大图 167s/页)
- **大 Excel 用 LibreOffice 中间转换**: `OfficeConverter` 对 `.xlsx`/`.xls` >5MB 自动走 `soffice --headless --convert-to csv` 再交 MarkItDown，远比 openpyxl 快
- **转换超时已内置**: `ConverterCoordinator(convert_timeout=120.0)` 单文件超时自动跳过
- **内容截断前置**: `ConverterCoordinator(max_output_chars=100000)` 转换阶段就截断，避免超大 .md 写入磁盘
- **openpyxl 警告已禁用**: `warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")` 在 office.py 模块级执行