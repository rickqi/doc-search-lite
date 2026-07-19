# V2EX 发布文

## 标题

[分享] 做了个本地文档搜索工具，不用向量数据库，纯 BM25 + LLM

## 正文

做保险行业的，手头几千份 PDF/DOCX（条款、诊断手册、监管文件），之前的痛点很直接：

- 按文件名搜 = 大海捞针
- 向量数据库贵还不好调，中文 embedding 效果飘忽
- 文档发到第三方 API 怕合规出问题

于是自己搓了一个全本地的文档智能搜索系统，开源了。

### 核心管线

```
源文件（PDF/DOCX/XLSX/PPTX/图片/邮件…）
    ↓ 自动转 Markdown
Tantivy BM25 全文索引（Rust 引擎，快）
    ↓
LLM Agent 自主搜索 → 阅读 → 回答
```

### 特色

**不用向量数据库，不用 GPU，不把文档发给第三方。** 纯 BM25 + jieba 分词 + Bigram 回退，对保险/法律/医疗这种专有名词多的场景，BM25 的关键词精确匹配比 embedding 靠谱太多。

**支持 11 种格式**：PDF、DOCX、XLSX、PPTX、HTML、CSV、纯文本、图片 OCR（4 种引擎）、Outlook .msg、ZIP/7z/RAR 压缩包自动解压

**三种使用方式**：
- `pip install` → CLI 三行命令零到搜索
- `python -m src.api` → 浏览器 Web UI（SSE 流式推送 Agent 搜索过程）
- MCP Server → OpenCode / Claude Desktop 直接调用

**Agent 做了不少优化**（参考了 NYU COMPILOT 论文）：
- ReAct 结构化推理，类似 o1 的思考链
- Draft 验证闭环 — 答案有依据才输出
- 工具反馈信号 — 搜不到自动换策略
- Best-of-K + 置信度校准

### 日常使用感受

问"年假怎么申请"，Agent 会自动搜索相关制度文档 → 阅读全文 → 定位具体条款 → 给出带原文引用的回答。全程不出本机，不花 API token（用的 DeepSeek V4 Flash，一次查询 ~$0.005）。

### Repo

https://github.com/rickqi/doc-search-lite

MIT 协议，欢迎 Star / PR / 吐槽。
