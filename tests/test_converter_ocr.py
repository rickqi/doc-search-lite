"""Tests for OCR service using GLM-4V."""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.converter.ocr import (
    OCRError,
    OCRErrorCode,
    OCRMode,
    OCRResult,
    OCRService,
    OCRServiceConfig,
    PaddleOCRHTTPService,
    get_paddleocr_http_service,
)


@pytest.fixture
def ocr_config():
    """Create test OCR configuration."""
    return OCRServiceConfig(
        api_key="test-api-key",
        base_url="https://test.api.url",
        max_retries=2,
        retry_delay=0.1,
    )


@pytest.fixture
def ocr_service(ocr_config):
    """Create OCR service instance."""
    return OCRService(ocr_config)


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample image file for testing."""
    # Create a minimal PNG file (1x1 pixel, black)
    image_content = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(image_content)
    return image_path


class TestOCRResult:
    """Tests for OCRResult dataclass."""

    def test_default_values(self):
        """Test default values of OCRResult."""
        result = OCRResult(success=True)
        assert result.success is True
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.processing_time == 0.0
        assert result.error is None
        assert result.error_code is None
        assert result.pages == 1
        assert result.model == ""
        assert result.source == ""

    def test_custom_values(self):
        """Test custom values of OCRResult."""
        result = OCRResult(
            success=True,
            text="Hello World",
            confidence=0.95,
            processing_time=1.5,
            model="glm-4v",
            source="test.png",
        )
        assert result.success is True
        assert result.text == "Hello World"
        assert result.confidence == 0.95
        assert result.processing_time == 1.5
        assert result.model == "glm-4v"
        assert result.source == "test.png"


class TestOCRError:
    """Tests for OCRError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = OCRError("Test error")
        assert str(error) == "Test error"
        assert error.code == OCRErrorCode.UNKNOWN
        assert error.original_error is None

    def test_error_with_code(self):
        """Test error with specific code."""
        error = OCRError("API error", code=OCRErrorCode.INVALID_API_KEY)
        assert error.code == OCRErrorCode.INVALID_API_KEY

    def test_error_with_original(self):
        """Test error with original exception."""
        original = ValueError("Original error")
        error = OCRError("Wrapped error", original_error=original)
        assert error.original_error == original


class TestOCRServiceConfig:
    """Tests for OCRServiceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OCRServiceConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.base_url is None
        assert config.model == "glm-ocr"
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.retry_multiplier == 2.0
        assert config.timeout == 60.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OCRServiceConfig(
            api_key="test-key",
            base_url="https://custom.url",
            model="custom-model",
            max_retries=5,
            retry_delay=0.5,
        )
        assert config.base_url == "https://custom.url"
        assert config.model == "custom-model"
        assert config.max_retries == 5
        assert config.retry_delay == 0.5

    def test_from_config(self):
        """Test creating from application Config."""
        from src.utils.config import Config

        app_config = Config(
            glm_api_key="app-api-key",
            glm_base_url="https://app.url",
        )
        ocr_config = OCRServiceConfig.from_config(app_config)
        assert ocr_config.api_key == "app-api-key"
        assert ocr_config.base_url == "https://app.url"


class TestOCRService:
    """Tests for OCRService."""

    def test_init(self, ocr_config):
        """Test service initialization."""
        service = OCRService(ocr_config)
        assert service._config == ocr_config
        assert service._client is None

    def test_is_url_true(self, ocr_service):
        """Test URL detection for valid URLs."""
        assert ocr_service._is_url("https://example.com/image.jpg") is True
        assert ocr_service._is_url("http://test.com/path/to/image.png") is True

    def test_is_url_false(self, ocr_service):
        """Test URL detection for non-URLs."""
        assert ocr_service._is_url("/path/to/image.jpg") is False
        assert ocr_service._is_url("image.png") is False
        assert ocr_service._is_url("C:\\path\\image.jpg") is False

    def test_get_mime_type(self, ocr_service):
        """Test MIME type detection."""
        assert ocr_service._get_mime_type(Path("image.jpg")) == "image/jpeg"
        assert ocr_service._get_mime_type(Path("image.png")) == "image/png"
        assert ocr_service._get_mime_type(Path("image.gif")) == "image/gif"
        assert ocr_service._get_mime_type(Path("image.webp")) == "image/webp"
        assert ocr_service._get_mime_type(Path("image.unknown")) == "image/jpeg"

    def test_encode_image(self, ocr_service, sample_image_path):
        """Test image encoding to base64."""
        encoded = ocr_service._encode_image(sample_image_path)
        assert isinstance(encoded, str)
        # Verify it's valid base64
        base64.b64decode(encoded)

    def test_encode_image_file_not_found(self, ocr_service):
        """Test encoding non-existent file."""
        with pytest.raises(OCRError) as exc_info:
            ocr_service._encode_image(Path("/nonexistent/image.jpg"))
        assert exc_info.value.code == OCRErrorCode.FILE_NOT_FOUND

    def test_classify_error_api_key(self, ocr_service):
        """Test error classification for API key errors."""
        error = Exception("Invalid api_key provided")
        assert ocr_service._classify_error(error) == OCRErrorCode.INVALID_API_KEY

        error = Exception("Authentication failed")
        assert ocr_service._classify_error(error) == OCRErrorCode.INVALID_API_KEY

    def test_classify_error_rate_limit(self, ocr_service):
        """Test error classification for rate limiting."""
        error = Exception("Rate limit exceeded")
        assert ocr_service._classify_error(error) == OCRErrorCode.RATE_LIMITED

        error = Exception("Quota exceeded")
        assert ocr_service._classify_error(error) == OCRErrorCode.RATE_LIMITED

    def test_classify_error_network(self, ocr_service):
        """Test error classification for network errors."""
        error = Exception("Connection timeout")
        assert ocr_service._classify_error(error) == OCRErrorCode.NETWORK_ERROR

        error = Exception("Network error")
        assert ocr_service._classify_error(error) == OCRErrorCode.NETWORK_ERROR

    def test_classify_error_invalid_image(self, ocr_service):
        """Test error classification for invalid image."""
        error = Exception("Invalid image format")
        assert ocr_service._classify_error(error) == OCRErrorCode.INVALID_IMAGE

    def test_classify_error_unknown(self, ocr_service):
        """Test error classification for unknown errors."""
        error = Exception("Something went wrong")
        assert ocr_service._classify_error(error) == OCRErrorCode.API_ERROR


class TestOCRServiceRecognize:
    """Tests for OCRService.recognize method."""

    @patch("src.converter.ocr.OCRService._get_client")
    def test_recognize_file_success(
        self, mock_get_client, ocr_service, sample_image_path
    ):
        """Test successful file recognition."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.layout_parsing.create.return_value = "Extracted text from image"
        mock_get_client.return_value = mock_client

        # Execute
        result = ocr_service.recognize(sample_image_path)

        # Verify
        assert result.success is True
        assert result.text == "Extracted text from image"
        assert result.error is None
        assert result.model == "glm-ocr"
        assert result.processing_time >= 0

    @patch("src.converter.ocr.OCRService._get_client")
    def test_recognize_url_success(self, mock_get_client, ocr_service):
        """Test successful URL recognition."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.layout_parsing.create.return_value = "Text from URL image"
        mock_get_client.return_value = mock_client

        # Execute
        result = ocr_service.recognize("https://example.com/test.jpg")

        # Verify
        assert result.success is True
        assert result.text == "Text from URL image"

    def test_recognize_file_not_found(self, ocr_service):
        """Test recognition with non-existent file."""
        result = ocr_service.recognize("/nonexistent/file.jpg")

        assert result.success is False
        assert result.error_code == OCRErrorCode.FILE_NOT_FOUND

    @patch("src.converter.ocr.OCRService._get_client")
    def test_recognize_api_error_invalid_key(
        self, mock_get_client, ocr_service, sample_image_path
    ):
        """Test recognition with invalid API key."""
        # Setup mock to raise error
        mock_client = MagicMock()
        mock_client.layout_parsing.create.side_effect = Exception("Invalid api_key")
        mock_get_client.return_value = mock_client

        # Execute
        result = ocr_service.recognize(sample_image_path)

        # Verify
        assert result.success is False
        assert result.error_code == OCRErrorCode.INVALID_API_KEY

    @patch("src.converter.ocr.OCRService._get_client")
    def test_recognize_retry_success(
        self, mock_get_client, ocr_service, sample_image_path
    ):
        """Test successful recognition after retry."""
        # Setup mock to fail once then succeed
        mock_client = MagicMock()
        mock_client.layout_parsing.create.side_effect = [
            Exception("Network error"),
            "Success after retry",
        ]
        mock_get_client.return_value = mock_client

        # Execute
        result = ocr_service.recognize(sample_image_path)

        # Verify
        assert result.success is True
        assert result.text == "Success after retry"
        assert mock_client.layout_parsing.create.call_count == 2

    @patch("src.converter.ocr.OCRService._get_client")
    def test_recognize_all_retries_fail(
        self, mock_get_client, ocr_service, sample_image_path
    ):
        """Test recognition failing after all retries."""
        # Setup mock to always fail
        mock_client = MagicMock()
        mock_client.layout_parsing.create.side_effect = Exception("Network timeout")
        mock_get_client.return_value = mock_client

        # Execute
        result = ocr_service.recognize(sample_image_path)

        # Verify
        assert result.success is False
        assert "failed after" in result.error.lower()
        assert (
            mock_client.layout_parsing.create.call_count
            == ocr_service._config.max_retries
        )

    @patch("src.converter.ocr.OCRService._get_client")
    def test_recognize_empty_response(
        self, mock_get_client, ocr_service, sample_image_path
    ):
        """Test recognition with empty API response."""
        # Setup mock with empty content
        mock_client = MagicMock()
        mock_client.layout_parsing.create.return_value = ""
        mock_get_client.return_value = mock_client

        # Execute
        result = ocr_service.recognize(sample_image_path)

        # Verify
        assert result.success is True
        assert result.text == ""


class TestOCRServiceRecognizeBatch:
    """Tests for OCRService.recognize_batch method."""

    @patch("src.converter.ocr.OCRService._get_client")
    def test_recognize_batch(self, mock_get_client, ocr_service, sample_image_path):
        """Test batch recognition."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.layout_parsing.create.return_value = "Batch text"
        mock_get_client.return_value = mock_client

        # Execute
        results = ocr_service.recognize_batch(
            [
                sample_image_path,
                "https://example.com/image.jpg",
            ]
        )

        # Verify
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_recognize_batch_mixed_results(self, ocr_service, sample_image_path):
        """Test batch recognition with mixed results."""
        # Execute with one valid and one invalid source
        results = ocr_service.recognize_batch(
            [
                sample_image_path,
                "/nonexistent/file.jpg",
            ]
        )

        # Verify - second one should fail immediately without API call
        assert len(results) == 2


class TestOCRServiceImport:
    """Test module imports."""

    def test_import_ocr_module(self):
        """Verify OCR module can be imported."""
        from src.converter import ocr

        assert ocr is not None

    def test_import_classes(self):
        """Verify all classes can be imported."""
        from src.converter.ocr import (
            OCRError,
            OCRErrorCode,
            OCRMode,
            OCRResult,
            OCRService,
            OCRServiceConfig,
        )

        assert OCRError is not None
        assert OCRErrorCode is not None
        assert OCRMode is not None
        assert OCRResult is not None
        assert OCRService is not None
        assert OCRServiceConfig is not None


class TestOCRMode:
    """Tests for OCRMode enum."""

    def test_layout_parsing_value(self):
        """Test LAYOUT_PARSING enum value."""
        assert OCRMode.LAYOUT_PARSING.value == "layout_parsing"

    def test_vision_chat_value(self):
        """Test VISION_CHAT enum value."""
        assert OCRMode.VISION_CHAT.value == "vision_chat"

    def test_enum_members(self):
        """Test all enum members exist."""
        members = list(OCRMode)
        assert len(members) == 2
        assert OCRMode.LAYOUT_PARSING in members
        assert OCRMode.VISION_CHAT in members


class TestOCRServiceConfigVisionMode:
    """Tests for OCRServiceConfig with vision mode."""

    def test_default_mode_is_layout_parsing(self):
        """Default mode is LAYOUT_PARSING."""
        config = OCRServiceConfig(api_key="test-key")
        assert config.mode == OCRMode.LAYOUT_PARSING

    def test_vision_mode_config(self):
        """Can create config with VISION_CHAT mode."""
        config = OCRServiceConfig(
            api_key="test-key",
            mode=OCRMode.VISION_CHAT,
        )
        assert config.mode == OCRMode.VISION_CHAT

    def test_vision_model_default(self):
        """Default vision model is glm-5-turbo."""
        config = OCRServiceConfig(api_key="test-key")
        assert config.vision_model == "glm-5-turbo"

    def test_custom_vision_model(self):
        """Can set custom vision model."""
        config = OCRServiceConfig(
            api_key="test-key",
            vision_model="glm-4v",
        )
        assert config.vision_model == "glm-4v"

    def test_vision_prompt_default(self):
        """Default vision prompt contains structured extraction rules."""
        config = OCRServiceConfig(api_key="test-key")
        assert "Markdown" in config.vision_prompt
        assert "表格" in config.vision_prompt

    def test_custom_vision_prompt(self):
        """Can set custom vision prompt."""
        config = OCRServiceConfig(
            api_key="test-key",
            vision_prompt="Custom prompt",
        )
        assert config.vision_prompt == "Custom prompt"


class TestOCRServiceVisionChat:
    """Tests for OCRService VISION_CHAT mode (mocked litellm)."""

    @pytest.fixture
    def vision_config(self):
        """Create OCR config in VISION_CHAT mode."""
        return OCRServiceConfig(
            api_key="test-api-key",
            base_url="https://test.api.url",
            mode=OCRMode.VISION_CHAT,
            max_retries=2,
            retry_delay=0.01,
            timeout=10.0,
        )

    @pytest.fixture
    def vision_service(self, vision_config):
        """Create OCR service in VISION_CHAT mode."""
        return OCRService(vision_config)

    @patch("src.converter.ocr.OCRService._call_vision_api")
    def test_vision_mode_dispatches_to_vision_api(
        self, mock_vision, vision_service, sample_image_path
    ):
        """VISION_CHAT mode calls _call_vision_api instead of layout_parsing."""
        mock_vision.return_value = ("Vision extracted text", {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
        result = vision_service.recognize(sample_image_path)
        assert result.success is True
        assert result.text == "Vision extracted text"
        mock_vision.assert_called_once()

    @patch("src.converter.ocr.OCRService._call_vision_api")
    def test_vision_mode_token_usage(
        self, mock_vision, vision_service, sample_image_path
    ):
        """VISION_CHAT mode returns token usage from litellm."""
        mock_vision.return_value = ("Text", {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280})
        result = vision_service.recognize(sample_image_path)
        assert result.token_usage["input_tokens"] == 200
        assert result.token_usage["output_tokens"] == 80

    @patch("litellm.completion")
    def test_call_vision_api_success(
        self, mock_completion, vision_service, sample_image_path
    ):
        """_call_vision_api calls litellm.completion with correct params."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Extracted via vision"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 500
        mock_usage.completion_tokens = 100
        mock_usage.total_tokens = 600
        mock_response.usage = mock_usage
        mock_completion.return_value = mock_response

        text, usage = vision_service._call_vision_api(sample_image_path)

        assert text == "Extracted via vision"
        assert usage["input_tokens"] == 500
        assert usage["output_tokens"] == 100
        assert usage["total_tokens"] == 600

        # Verify litellm.completion was called with correct model prefix
        call_kwargs = mock_completion.call_args
        assert call_kwargs[1]["model"] == "zai/glm-5-turbo"

    @patch("litellm.completion")
    def test_call_vision_api_retry(
        self, mock_completion, vision_service, sample_image_path
    ):
        """_call_vision_api retries on network error."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Success after retry"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 0
        mock_usage.completion_tokens = 0
        mock_usage.total_tokens = 0
        mock_response.usage = mock_usage

        mock_completion.side_effect = [
            Exception("Network error"),
            mock_response,
        ]

        text, usage = vision_service._call_vision_api(sample_image_path)
        assert text == "Success after retry"
        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    def test_call_vision_api_all_retries_fail(
        self, mock_completion, vision_service, sample_image_path
    ):
        """_call_vision_api raises OCRError after all retries fail."""
        mock_completion.side_effect = Exception("Network timeout")

        with pytest.raises(OCRError) as exc_info:
            vision_service._call_vision_api(sample_image_path)

        assert "failed after" in str(exc_info.value).lower()

    @patch("litellm.completion")
    def test_call_vision_api_no_retry_on_auth_error(
        self, mock_completion, vision_service, sample_image_path
    ):
        """_call_vision_api does not retry on auth errors."""
        mock_completion.side_effect = Exception("Invalid api_key")

        with pytest.raises(OCRError) as exc_info:
            vision_service._call_vision_api(sample_image_path)

        assert exc_info.value.code == OCRErrorCode.INVALID_API_KEY
        assert mock_completion.call_count == 1

    @patch("litellm.completion")
    def test_call_vision_api_url_input(self, mock_completion, vision_service):
        """_call_vision_api handles URL input without file read."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "URL text"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 0
        mock_usage.completion_tokens = 0
        mock_usage.total_tokens = 0
        mock_response.usage = mock_usage
        mock_completion.return_value = mock_response

        text, usage = vision_service._call_vision_api("https://example.com/img.jpg")
        assert text == "URL text"

    def test_call_vision_api_file_not_found(self, vision_service):
        """_call_vision_api raises on missing local file."""
        with pytest.raises(OCRError) as exc_info:
            vision_service._call_vision_api(Path("/nonexistent/img.png"))
        assert exc_info.value.code == OCRErrorCode.FILE_NOT_FOUND

    @patch("src.converter.ocr.OCRService._call_layout_parsing_api")
    def test_layout_parsing_mode_unchanged(
        self, mock_layout, ocr_service, sample_image_path
    ):
        """LAYOUT_PARSING mode still calls the existing API path."""
        mock_layout.return_value = ("Layout text", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        result = ocr_service.recognize(sample_image_path)
        assert result.success is True
        assert result.text == "Layout text"
        mock_layout.assert_called_once()


class TestOCRServicePrepareFileParam:
    """Tests for _prepare_file_param helper."""

    def test_path_input(self, ocr_service, sample_image_path):
        """Path input returns data URL."""
        result = ocr_service._prepare_file_param(sample_image_path)
        assert result.startswith("data:image/png;base64,")

    def test_url_input(self, ocr_service):
        """URL input is passed through."""
        url = "https://example.com/img.jpg"
        result = ocr_service._prepare_file_param(url)
        assert result == url

    def test_nonexistent_file(self, ocr_service):
        """Non-existent local file raises OCRError."""
        with pytest.raises(OCRError) as exc_info:
            ocr_service._prepare_file_param("/nonexistent/file.png")
        assert exc_info.value.code == OCRErrorCode.FILE_NOT_FOUND


# ═══════════════════════════════════════════════════════════════
# PaddleOCRHTTPService
# ═══════════════════════════════════════════════════════════════

import json as _json


class TestPaddleOCRHTTPService:
    """Tests for PaddleOCRHTTPService (remote HTTP OCR)."""

    @pytest.fixture
    def service(self):
        return PaddleOCRHTTPService(base_url="http://localhost:8868")

    @pytest.fixture
    def sample_image_path(self, tmp_path):
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return image_path

    def test_default_url_from_env(self, monkeypatch):
        monkeypatch.setenv("PADDLEOCR_HTTP_URL", "http://192.168.1.100:9999")
        svc = PaddleOCRHTTPService()
        assert svc.base_url == "http://192.168.1.100:9999"

    def test_default_url_fallback(self, monkeypatch):
        monkeypatch.delenv("PADDLEOCR_HTTP_URL", raising=False)
        svc = PaddleOCRHTTPService()
        assert svc.base_url == "http://localhost:8868"

    def test_explicit_url_overrides_env(self, monkeypatch):
        monkeypatch.setenv("PADDLEOCR_HTTP_URL", "http://wrong:9999")
        svc = PaddleOCRHTTPService(base_url="http://right:8868")
        assert svc.base_url == "http://right:8868"

    def test_health_ok(self, service):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"status":"ok","gpu":true}'
        mock_resp.__enter__.return_value = mock_resp

        with patch("src.converter.ocr.urlopen", return_value=mock_resp):
            assert service.health() is True

    def test_health_down(self, service):
        with patch("src.converter.ocr.urlopen", side_effect=OSError("Connection refused")):
            assert service.health() is False

    def test_health_unexpected_response(self, service):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"status":"error"}'
        mock_resp.__enter__.return_value = mock_resp

        with patch("src.converter.ocr.urlopen", return_value=mock_resp):
            assert service.health() is False

    def test_recognize_success(self, service, sample_image_path):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _json.dumps({
            "texts": [
                {"text": "测试文本一", "confidence": 0.99},
                {"text": "测试文本二", "confidence": 0.85},
            ],
            "count": 2,
            "time_ms": 268,
        }).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch("src.converter.ocr.urlopen", return_value=mock_resp):
            result = service.recognize(sample_image_path)

        assert result.success is True
        assert "测试文本一" in result.text
        assert "测试文本二" in result.text
        assert result.model == "PP-OCRv5-HTTP"
        assert 0.8 < result.confidence < 1.0
        assert result.source == str(sample_image_path)

    def test_recognize_empty_results(self, service, sample_image_path):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _json.dumps({
            "texts": [],
            "count": 0,
            "time_ms": 50,
        }).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch("src.converter.ocr.urlopen", return_value=mock_resp):
            result = service.recognize(sample_image_path)

        assert result.success is True
        assert result.text == ""
        assert result.confidence == 0.0

    def test_recognize_file_not_found(self, service):
        result = service.recognize(Path("/nonexistent/image.png"))
        assert result.success is False
        assert result.error_code == OCRErrorCode.FILE_NOT_FOUND

    def test_recognize_server_error(self, service, sample_image_path):
        with patch("src.converter.ocr.urlopen", side_effect=OSError("Timeout")):
            result = service.recognize(sample_image_path)

        assert result.success is False
        assert result.error_code == OCRErrorCode.NETWORK_ERROR
        assert "Timeout" in result.error

    def test_multipart_encoding(self, service, sample_image_path):
        content_type, body = service._encode_multipart(sample_image_path)

        assert "multipart/form-data" in content_type
        assert b"boundary=" in content_type.encode()
        assert sample_image_path.name.encode() in body
        assert b'form-data; name="file"' in body

    def test_recognize_batch(self, service, sample_image_path):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _json.dumps({
            "texts": [{"text": "测试", "confidence": 0.99}],
            "count": 1,
            "time_ms": 100,
        }).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch("src.converter.ocr.urlopen", return_value=mock_resp):
            results = service.recognize_batch([sample_image_path, sample_image_path])

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_base_url_trailing_slash_handled(self):
        svc = PaddleOCRHTTPService(base_url="http://localhost:8868/")
        assert svc.base_url == "http://localhost:8868"


# ═══════════════════════════════════════════════════════════════
# PaddleOCRHTTPService Singleton
# ═══════════════════════════════════════════════════════════════


class TestPaddleOCRHTTPServiceSingleton:
    """Tests for singleton factory function."""

    def test_singleton_returns_same_instance(self, monkeypatch):
        monkeypatch.delenv("PADDLEOCR_HTTP_URL", raising=False)

        import src.converter.ocr as mod
        mod._paddleocr_http_service = None

        svc1 = get_paddleocr_http_service()
        svc2 = get_paddleocr_http_service()
        assert svc1 is svc2
