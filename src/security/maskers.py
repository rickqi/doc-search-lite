"""Maskers — PII/Keyword/Regex 掩码器."""

from __future__ import annotations

import abc
import re

from src.security.desensitizer import DesensitizeResult


class BaseMasker(abc.ABC):
    """掩码器基类."""

    @abc.abstractmethod
    def process(self, result: DesensitizeResult) -> DesensitizeResult:
        """对文本执行脱敏，返回更新后的结果."""
        ...

    def _mask_pattern(self, result: DesensitizeResult, type_name: str, pattern: str) -> DesensitizeResult:
        """通用正则替换逻辑: 找到所有匹配 → 替换为占位符 → 记录映射."""
        compiled = re.compile(pattern)
        text = result.masked_text
        mapping = dict(result.mapping)
        counter = result.counter

        def _replacer(m: re.Match) -> str:
            nonlocal counter
            placeholder = f"[{type_name.upper()}_{counter}]"
            counter += 1
            mapping[placeholder] = m.group(0)
            return placeholder

        new_text = compiled.sub(_replacer, text)
        return DesensitizeResult(masked_text=new_text, mapping=mapping, counter=counter)


class PIIMasker(BaseMasker):
    r"""PII 模式掩码 — 手机/身份证/邮箱/IP/银行卡.

    默认启用的类型:
      - phone: 1[3-9]\d{9}
      - id_card: \d{17}[\dXx]
      - bank_card: \d{16,19}

    默认禁用的类型 (可配置开启):
      - email: 正则, 可能误伤正常内容
      - ip: 正则, 可能误伤版本号/统计数字
    """

    # 高置信度模式 (默认启用)
    # 注意匹配顺序: 先匹配长模式 (身份证/银行卡), 再匹配短模式 (手机号)
    # 避免手机号模式误匹配身份证中的数字片段
    PATTERNS_HIGH_CONFIDENCE = {
        "id_card": r"(?<!\d)\d{17}[\dXx](?!\d)",  # 18位身份证, 前后无数字
        "bank_card": r"(?<!\d)\d{16,19}(?!\d)",    # 16-19位银行卡, 前后无数字
        "phone": r"(?<!\d)1[3-9]\d{9}(?!\d)",      # 11位手机号, 前后无数字
    }

    # 低置信度模式 (默认禁用, 可配置开启)
    PATTERNS_LOW_CONFIDENCE = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "ip": r"(?<!\d)(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?!\d)",
    }

    def __init__(self, enabled_types: dict[str, bool] | None = None):
        """初始化 PIIMasker.

        Args:
            enabled_types: {type_name: bool} 覆盖默认启停状态。
                          None = 使用默认 (phone/id_card/bank_card 启用)。
        """
        self.patterns: dict[str, str] = {}
        if enabled_types is None:
            self.patterns.update(self.PATTERNS_HIGH_CONFIDENCE)
        else:
            for name, pat in {**self.PATTERNS_HIGH_CONFIDENCE, **self.PATTERNS_LOW_CONFIDENCE}.items():
                if enabled_types.get(name, name in self.PATTERNS_HIGH_CONFIDENCE):
                    self.patterns[name] = pat

    def process(self, result: DesensitizeResult) -> DesensitizeResult:
        for type_name, pattern in self.patterns.items():
            result = self._mask_pattern(result, type_name, pattern)
        return result


class KeywordMasker(BaseMasker):
    """关键词掩码 — 从配置文件加载敏感词列表."""

    def __init__(self, keywords: list[str]):
        # 按长度降序排列，确保长词优先匹配，避免短词吞掉长词
        self.keywords = sorted(keywords, key=len, reverse=True)

    def process(self, result: DesensitizeResult) -> DesensitizeResult:
        text = result.masked_text
        mapping = dict(result.mapping)
        counter = result.counter

        for keyword in self.keywords:
            if keyword in text:
                placeholder = f"[KEYWORD_{counter}]"
                counter += 1
                text = text.replace(keyword, placeholder)
                mapping[placeholder] = keyword

        return DesensitizeResult(masked_text=text, mapping=mapping, counter=counter)


class RegexMasker(BaseMasker):
    """自定义正则掩码 — 从配置文件加载用户定义规则."""

    def __init__(self, rules: list[dict[str, str]]):
        """rules: [{"name": "保单号", "pattern": "[A-Z]{2}\\\\d{8,12}"}]"""
        self.rules = rules

    def process(self, result: DesensitizeResult) -> DesensitizeResult:
        for rule in self.rules:
            result = self._mask_pattern(result, rule.get("name", "custom"), rule["pattern"])
        return result
