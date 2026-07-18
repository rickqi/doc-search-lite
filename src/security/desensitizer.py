"""Desensitizer — 统一脱敏入口.

独立于 maskers, 提供 desensitize/restore 接口和 fail-safe 包装.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DesensitizeResult:
    """单次脱敏的结果.

    Attributes:
        masked_text: 脱敏后的文本.
        mapping: {占位符 -> 原始值} 映射表, 用于 restore.
        counter: 内部计数器, 确保占位符唯一.
    """

    masked_text: str = ""
    mapping: dict[str, str] = field(default_factory=dict)
    counter: int = 0


class Desensitizer:
    """统一脱敏入口 — 零侵入集成.

    用法::

        d = Desensitizer()
        result = d.desensitize("手机号 13800138000")
        print(result.masked_text)   # "手机号 [PHONE_0]"
        restored = d.restore(result.masked_text, result.mapping)
        print(restored)             # "手机号 13800138000"
    """

    def __init__(self, config_path: Path | None = None):

        self.maskers: list[Any] = []
        self._init_default_maskers()

    def _init_default_maskers(self):
        """初始化默认脱敏器 (PII)."""
        from src.security.maskers import PIIMasker
        self.maskers.append(PIIMasker())

    def desensitize(self, text: str) -> DesensitizeResult:
        """对文本进行脱敏处理.

        Args:
            text: 原文.

        Returns:
            DesensitizeResult(masked_text, mapping).
            异常时返回原文 + 空映射 (fail-safe).
        """
        if not text:
            return DesensitizeResult(masked_text=text or "")
        try:
            result = DesensitizeResult(masked_text=text)
            for masker in self.maskers:
                result = masker.process(result)
            return result
        except Exception as e:
            logger.warning("Desensitization failed, using original: %s", e)
            return DesensitizeResult(masked_text=text)

    def restore(self, masked_text: str, mapping: dict[str, str]) -> str:
        """将脱敏标记恢复为原始值.

        Args:
            masked_text: LLM 返回的可能含占位符的文本.
            mapping: desensitize 返回的映射表.

        Returns:
            恢复后的文本. 异常时原样返回 (fail-safe).
        """
        if not mapping or not masked_text:
            return masked_text
        try:
            result = masked_text
            for placeholder, original in mapping.items():
                result = result.replace(placeholder, original)
            return result
        except Exception as e:
            logger.warning("Restore failed, returning masked: %s", e)
            return masked_text
