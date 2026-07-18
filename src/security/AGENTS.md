# Security 模块

## OVERVIEW
PII 脱敏模块。LLM 调用前自动对文本进行脱敏处理，LLM 回答后自动恢复原始数据。零侵入集成（LLMClient 内部自动处理）。所有异常 fail-safe。

## STRUCTURE
```
security/
├── __init__.py        # 模块导出 (Desensitizer, PIIMasker, etc.)
├── desensitizer.py    # Desensitizer — 统一脱敏入口 (desensitize/restore) + fail-safe
└── maskers.py         # BaseMasker + PIIMasker + KeywordMasker + RegexMasker
```

## WHERE TO LOOK
| 任务 | 文件 | 关键 |
|------|------|------|
| 统一入口 | desensitizer.py | Desensitizer.desensitize(text) → DesensitizeResult (masked_text + mapping) |
| 恢复原文 | desensitizer.py | Desensitizer.restore(masked_text, mapping) → 原文 |
| 内置PII | maskers.py | PIIMasker — phone/id_card/bank_card(默认启用), email/IP(可配置) |
| 关键词 | maskers.py | KeywordMasker(keywords=[...]) — 可配置敏感词列表 |
| 自定义正则 | maskers.py | RegexMasker(rules=[{"name","pattern"}]) — 自定义模式 |
| 通用替换 | maskers.py | BaseMasker._mask_pattern() — 正则匹配→占位符替换→记录映射 |

## CONVENTIONS
- **脱敏标记格式**: `[TYPE_N]` (如 `[PHONE_0]`, `[ID_CARD_1]`)
- **匹配顺序**: id_card > bank_card > phone（长模式优先，防误匹配）
- **word boundary**: 所有正则使用 `(?<!\d)...(?!\d)` 前后断言，避免数字片段误匹配
- **默认启用**: phone / id_card / bank_card（高置信度）；email / IP 默认关闭
- **fail-safe**: 任何异常 → 原文发送，记录 warning 日志
- **环境变量**: `DESENSITIZE_ENABLED=true|false`（默认 true），`DESENSITIZE_CONFIG`（YAML 配置路径）
- **零侵入**: 自动在 `LLMClient.__init__()` 中初始化，`chat()` 中自动脱敏→恢复，无需修改调用点

## ANTI-PATTERNS
- 不要禁用 fail-safe — 脱敏失败必须回退到原文，不能阻断 LLM 调用
- 不要在脱敏后手动修改 mapping — 会导致 restore 无法正确恢复
- 不要在低置信度模式（email/IP）默认启用 — 可能误伤正常文本
- 不要对 tool/assistant 角色消息脱敏 — 只脱敏 user/system
