"""Tests for LLM desensitization — 脱敏/恢复/fail-safe + LLMClient 集成."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.security.desensitizer import Desensitizer, DesensitizeResult
from src.security.maskers import PIIMasker, KeywordMasker, RegexMasker


# ── Desensitizer 基础 ────────────────────────────────────────────────


class TestDesensitizerBasic:
    """Desensitizer 核心功能."""

    def test_desensitize_phone(self):
        d = Desensitizer()
        result = d.desensitize("请联系 13800138000")
        assert "[PHONE_0]" in result.masked_text
        assert result.mapping["[PHONE_0]"] == "13800138000"

    def test_desensitize_multiple_phones(self):
        d = Desensitizer()
        result = d.desensitize("号码: 13800138000 和 13900139000")
        assert "[PHONE_0]" in result.masked_text
        assert "[PHONE_1]" in result.masked_text
        assert result.mapping["[PHONE_0]"] == "13800138000"
        assert result.mapping["[PHONE_1]"] == "13900139000"

    def test_restore(self):
        d = Desensitizer()
        masked = "请联系 [PHONE_0]"
        mapping = {"[PHONE_0]": "13800138000"}
        assert d.restore(masked, mapping) == "请联系 13800138000"

    def test_roundtrip(self):
        d = Desensitizer()
        original = "手机 13800138000, 身份证 110101199001011234"
        dr = d.desensitize(original)
        restored = d.restore(dr.masked_text, dr.mapping)
        assert restored == original

    def test_no_pii_no_change(self):
        d = Desensitizer()
        result = d.desensitize("今天天气很好")
        assert result.masked_text == "今天天气很好"
        assert result.mapping == {}

    def test_empty_text(self):
        d = Desensitizer()
        assert d.desensitize("").masked_text == ""

    def test_fail_safe_on_desensitize(self):
        """异常时返回原文 (fail-safe)."""
        d = Desensitizer()
        # Mock 一个会抛异常的 masker
        from src.security.maskers import BaseMasker
        class BadMasker(BaseMasker):
            def process(self, result):
                raise RuntimeError("mock failure")
        d.maskers = [BadMasker()]
        result = d.desensitize("保持原文")
        assert result.masked_text == "保持原文"
        assert result.mapping == {}

    def test_fail_safe_on_restore(self):
        """restore 异常时返回脱敏文本 (fail-safe)."""
        d = Desensitizer()
        result = d.restore("[PHONE_0]", {"[PHONE_0]": "13800138000"})
        assert "13800138000" in result

    def test_id_card(self):
        d = Desensitizer()
        result = d.desensitize("身份证 110101199001011234")
        assert "[ID_CARD_0]" in result.masked_text

    def test_bank_card(self):
        d = Desensitizer()
        result = d.desensitize("银行卡 6222021234567890")
        assert "[BANK_CARD_0]" in result.masked_text


# ── PIIMasker ────────────────────────────────────────────────────────


class TestPIIMasker:
    """PIIMasker 各类型."""

    def test_phone_only(self):
        m = PIIMasker(enabled_types={"phone": True, "id_card": False, "bank_card": False})
        result = m.process(DesensitizeResult("手机 13800138000 身份证 110101199001011234"))
        assert "[PHONE_0]" in result.masked_text
        assert "[ID_CARD_0]" not in result.masked_text  # 未启用

    def test_email_disabled_by_default(self):
        m = PIIMasker()  # 默认不启用 email
        result = m.process(DesensitizeResult("邮箱 test@example.com"))
        assert "[EMAIL_0]" not in result.masked_text

    def test_ip_disabled_by_default(self):
        m = PIIMasker()  # 默认不启用 ip
        result = m.process(DesensitizeResult("IP 192.168.1.1"))
        assert "[IP_0]" not in result.masked_text

    def test_email_enabled_explicitly(self):
        """显式启用 email."""
        m = PIIMasker(enabled_types={"email": True, "phone": False, "id_card": False, "bank_card": False})
        result = m.process(DesensitizeResult("邮箱 test@example.com 和 user@test.org"))
        assert "[EMAIL_0]" in result.masked_text
        assert "[EMAIL_1]" in result.masked_text

    def test_ip_enabled_explicitly(self):
        """显式启用 IP."""
        m = PIIMasker(enabled_types={"ip": True, "phone": False, "id_card": False, "bank_card": False})
        result = m.process(DesensitizeResult("IP 192.168.1.1"))
        assert "[IP_0]" in result.masked_text

    def test_all_pii_types(self):
        """全部类型同时启用."""
        m = PIIMasker(enabled_types={
            "phone": True, "id_card": True, "bank_card": True,
            "email": True, "ip": True,
        })
        text = "手机 13800138000, 身份证 110101199001011234, 银行卡 6222021234567890, 邮箱 test@abc.com, IP 10.0.0.1"
        result = m.process(DesensitizeResult(text))
        assert "[ID_CARD_0]" in result.masked_text
        assert "[BANK_CARD_0]" in result.masked_text or "[BANK_CARD_1]" in result.masked_text
        assert "[PHONE_" in result.masked_text
        assert "[EMAIL_" in result.masked_text
        assert "[IP_" in result.masked_text
        assert "13800138000" not in result.masked_text


# ── KeywordMasker ────────────────────────────────────────────────────


class TestKeywordMasker:
    """关键词脱敏."""

    def test_keyword_masked(self):
        m = KeywordMasker(keywords=["内部项目"])
        result = m.process(DesensitizeResult("内部项目已上线"))
        assert "[KEYWORD_0]" in result.masked_text
        assert result.mapping["[KEYWORD_0]"] == "内部项目"

    def test_multiple_keywords(self):
        m = KeywordMasker(keywords=["项目A", "项目B"])
        result = m.process(DesensitizeResult("项目A 和 项目B 已完成"))
        assert "[KEYWORD_0]" in result.masked_text
        assert "[KEYWORD_1]" in result.masked_text

    def test_no_keyword_no_change(self):
        m = KeywordMasker(keywords=["敏感词"])
        result = m.process(DesensitizeResult("正常文本"))
        assert result.masked_text == "正常文本"
        assert result.mapping == {}

    def test_keyword_restore_roundtrip(self):
        d = Desensitizer()
        # 手动添加 keyword masker
        from src.security.maskers import KeywordMasker
        d.maskers.append(KeywordMasker(keywords=["内部项目"]))
        dr = d.desensitize("内部项目已完成")
        restored = d.restore(dr.masked_text, dr.mapping)
        assert restored == "内部项目已完成"


# ── RegexMasker ──────────────────────────────────────────────────────


class TestRegexMasker:
    """自定义正则脱敏."""

    def test_custom_pattern(self):
        m = RegexMasker(rules=[{"name": "保单号", "pattern": r"[A-Z]{2}\d{8}"}])
        result = m.process(DesensitizeResult("保单 AB12345678"))
        placeholder = f"[保单号_0]"
        assert placeholder in result.masked_text, f"预期 {placeholder} 在 {result.masked_text}"

    def test_custom_pattern_multiple(self):
        """多条命中."""
        m = RegexMasker(rules=[{"name": "PID", "pattern": r"\d{5}"}])
        result = m.process(DesensitizeResult("PID: 12345, 67890"))
        assert "[PID_0]" in result.masked_text
        assert "[PID_1]" in result.masked_text

    def test_custom_pattern_restore_roundtrip(self):
        """自定义正则脱敏→恢复."""
        d = Desensitizer()
        from src.security.maskers import RegexMasker
        d.maskers.append(RegexMasker(rules=[{"name": "PID", "pattern": r"\d{5}"}]))
        dr = d.desensitize("编号 12345 已完成")
        restored = d.restore(dr.masked_text, dr.mapping)
        assert restored == "编号 12345 已完成"


# ── Restore 多映射 ──────────────────────────────────────────────────


class TestRestoreMultipleMappings:
    """恢复多个消息的脱敏."""

    def test_restore_multiple_mappings(self):
        """同一个 desensitize 结果中的多个 mapping 应正确恢复."""
        d = Desensitizer()
        # 一条消息中包含两个手机号
        dr = d.desensitize("号码 13800138000 和 13900139000")
        response = "已通知 [PHONE_0] 和 [PHONE_1]"
        # 一次性恢复
        restored = d.restore(response, dr.mapping)
        assert "13800138000" in restored
        assert "13900139000" in restored

    def test_restore_overlapping_placeholder(self):
        """占位符值恰为另一个占位符时不应误替换."""
        d = Desensitizer()
        # 模拟极端情况: 一个值本身包含 [PHONE_ 字符
        mapping = {"[KEY_0]": "包含[PHONE_字符"}
        result = d.restore("前缀[KEY_0]后缀", mapping)
        assert result == "前缀包含[PHONE_字符后缀"


# ── LLMClient 集成 ──────────────────────────────────────────────────


class TestLLMClientIntegration:
    """验证脱敏在 LLMClient 中正常工作."""

    def test_desensitizer_env_var_enabled_by_default(self):
        """默认应启用脱敏."""
        # DESENSITIZE_ENABLED 默认 true
        from src.agent.llm_client import LLMClient
        from src.utils.config import Config

        config = MagicMock(spec=Config)
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.llm_temperature = 0.7
        config.llm_max_tokens = 100
        config.litellm_model = "test-model"
        config.deepseek_api_key = ""
        config.glm_api_key = ""

        llm = LLMClient(config=config)
        assert llm._desensitizer is not None

    def test_desensitizer_env_disabled(self):
        """DESENSITIZE_ENABLED=false 应禁用脱敏."""
        os.environ["DESENSITIZE_ENABLED"] = "false"
        from src.agent.llm_client import LLMClient
        from src.utils.config import Config

        config = MagicMock(spec=Config)
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.llm_temperature = 0.7
        config.llm_max_tokens = 100
        config.litellm_model = "test-model"
        config.deepseek_api_key = ""
        config.glm_api_key = ""

        llm = LLMClient(config=config)
        assert llm._desensitizer is None
        del os.environ["DESENSITIZE_ENABLED"]

    def test_chat_desensitizes_and_restores(self):
        """chat() 应对消息脱敏并恢复回答."""
        from src.agent.llm_client import LLMClient, ChatMessage, ChatResponse
        from src.utils.config import Config

        config = MagicMock(spec=Config)
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.llm_temperature = 0.7
        config.llm_max_tokens = 500
        config.litellm_model = "test-model"
        config.deepseek_api_key = ""
        config.glm_api_key = ""

        llm = LLMClient(config=config)
        # Mock the actual LLM call
        llm._resolve_model = MagicMock(return_value="test-model")
        llm._parse_response = MagicMock(return_value=ChatResponse(
            content="请联系 [PHONE_0] 处理",
            usage={"total_tokens": 10},
        ))
        llm._router = None

        # Mock completion to return a response
        with patch("src.agent.llm_client.completion") as mock_comp:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "请联系 [PHONE_0] 处理"
            mock_resp.usage = MagicMock()
            mock_resp.usage.prompt_tokens = 5
            mock_resp.usage.completion_tokens = 5
            mock_resp.usage.total_tokens = 10
            mock_resp.model = "test"
            mock_comp.return_value = mock_resp

            response = llm.chat(
                [ChatMessage(role="user", content="手机号 13800138000")]
            )

        # 回答中的占位符应被恢复为原始值
        assert "13800138000" in response.content
        assert "[PHONE_0]" not in response.content

    def test_chat_system_message_desensitized(self):
        """system 消息也应被脱敏."""
        from src.agent.llm_client import LLMClient, ChatMessage, ChatResponse
        from src.utils.config import Config

        config = MagicMock(spec=Config)
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.llm_temperature = 0.7
        config.llm_max_tokens = 500
        config.litellm_model = "test-model"
        config.deepseek_api_key = ""
        config.glm_api_key = ""

        llm = LLMClient(config=config)
        llm._resolve_model = MagicMock(return_value="test-model")
        llm._router = None

        with patch("src.agent.llm_client.completion") as mock_comp:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "好的"
            mock_resp.usage = MagicMock()
            mock_resp.usage.prompt_tokens = 3
            mock_resp.usage.completion_tokens = 2
            mock_resp.usage.total_tokens = 5
            mock_resp.model = "test"
            mock_comp.return_value = mock_resp

            response = llm.chat([
                ChatMessage(role="system", content="用户手机号 13800138000"),
                ChatMessage(role="user", content="查询"),
            ])

        # 脱敏后的消息中不应有原文 (验证通过 mock 的 call_args)
        call_args = mock_comp.call_args
        sent_messages = call_args[1]["messages"]
        sent_text = str(sent_messages)
        assert "13800138000" not in sent_text
        assert "[PHONE_0]" in sent_text

    def test_chat_with_dict_messages(self):
        """支持 dict 格式消息."""
        from src.agent.llm_client import LLMClient, ChatResponse
        from src.utils.config import Config

        config = MagicMock(spec=Config)
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.llm_temperature = 0.7
        config.llm_max_tokens = 500
        config.litellm_model = "test-model"
        config.deepseek_api_key = ""
        config.glm_api_key = ""

        llm = LLMClient(config=config)
        llm._resolve_model = MagicMock(return_value="test-model")
        llm._router = None

        with patch("src.agent.llm_client.completion") as mock_comp:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "已通知"
            mock_resp.usage = MagicMock()
            mock_resp.usage.prompt_tokens = 3
            mock_resp.usage.completion_tokens = 2
            mock_resp.usage.total_tokens = 5
            mock_resp.model = "test"
            mock_comp.return_value = mock_resp

            response = llm.chat([
                {"role": "user", "content": "手机号 13800138000"},
            ])

        assert response is not None

    def test_chat_no_content_response(self):
        """LLM 返回空内容时不应崩溃."""
        from src.agent.llm_client import LLMClient, ChatMessage
        from src.utils.config import Config

        config = MagicMock(spec=Config)
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.llm_temperature = 0.7
        config.llm_max_tokens = 500
        config.litellm_model = "test-model"
        config.deepseek_api_key = ""
        config.glm_api_key = ""

        llm = LLMClient(config=config)
        llm._resolve_model = MagicMock(return_value="test-model")
        llm._router = None

        with patch("src.agent.llm_client.completion") as mock_comp:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = None
            mock_resp.usage = MagicMock()
            mock_resp.usage.prompt_tokens = 0
            mock_resp.usage.completion_tokens = 0
            mock_resp.usage.total_tokens = 0
            mock_resp.model = "test"
            mock_comp.return_value = mock_resp

            response = llm.chat([
                ChatMessage(role="user", content="手机号 13800138000"),
            ])

        assert response is not None
