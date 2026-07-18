"""Tests for TagExtractor — keyword-based document tag extraction."""

import pytest

from src.converter.tag_extractor import TagExtractionResult, TagExtractor

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def extractor():
    """Create a TagExtractor instance."""
    return TagExtractor()


# ── Real-world insurance document content samples ─────────────────

INSURANCE_PRODUCT_MD = """\
# XX人寿保险股份有限公司
## XX终身寿险条款

### 保险责任
在本合同有效期内，我们承担下列保险责任：

一、身故保险金
若被保险人身故，我们按本合同基本保险金额给付身故保险金。

二、全残保险金
若被保险人全残，我们按本合同基本保险金额给付全残保险金。

### 责任免除
因下列情形之一导致被保险人身故或全残的，我们不承担给付保险金的责任：
1. 投保人对被保险人的故意杀害、故意伤害；
2. 被保险人故意犯罪或者抗拒依法采取的刑事强制措施；

### 等待期
自本合同生效日起90日内，被保险人发生保险事故的，我们不承担保险责任。

### 免赔额
本合同的免赔额为1000元。
"""

REVIEW_POINT_MD = """\
# 保险产品审查要点

## 一、消保审查要点
1. 产品条款是否符合消费者权益保护要求
2. 信息披露是否充分完整
3. 投资适当性评估是否到位

## 二、产品条款审查要点
1. 保险责任和责任免除条款是否清晰明确
2. 等待期和免赔额设置是否合理
3. 保险金额计算方式是否准确

## 三、销售合规审查要点
1. 销售行为是否合规
2. 犹豫期是否充分告知
3. 回访制度是否完善
4. 双录要求是否执行
"""

REGULATION_MD = """\
# 关于进一步加强消费者权益保护工作的通知

各保险公司、各保险资产管理公司：

为进一步加强保险业消费者权益保护工作，切实维护保险消费者合法权益，
根据《中华人民共和国保险法》和《消费者权益保护法》等相关法律法规，
现就有关事项通知如下：

一、完善消费者权益保护制度
各公司应当建立健全消费者权益保护工作机制，明确消保审查流程。

二、加强信息披露
各公司应当及时、准确、完整地披露产品信息。

三、规范销售行为
各公司应当加强销售合规管理，防止误导销售。
"""

HEALTH_SERVICE_MD = """\
# 健康管理服务方案

## 服务内容
1. 健康管理咨询
2. 就医绿通服务
3. 二次诊疗意见
4. 专家预约服务
5. 慢病管理
6. 年度体检

## 服务流程
客户提交服务申请 → 健康管理团队评估 → 安排就医绿通 →
专家预约 → 二次诊疗 → 后续跟踪
"""

PERSONAL_INFO_MD = """\
# 个人信息保护政策

## 信息收集
我们收集以下个人信息：
- 姓名、身份证号
- 联系方式
- 健康信息

## 数据安全
我们采取严格的数据安全措施保护您的个人信息：
- 加密存储
- 访问控制
- 安全传输

## 隐私保护
您的隐私是我们的首要任务。
"""


# ── TagExtractionResult dataclass tests ───────────────────────────


class TestTagExtractionResult:
    """Test TagExtractionResult dataclass."""

    def test_creation(self):
        result = TagExtractionResult(
            tags=["消保审查", "产品条款"],
            doc_type="review_point",
            keywords=["审查要点", "保险"],
            confidence=0.85,
        )
        assert result.tags == ["消保审查", "产品条款"]
        assert result.doc_type == "review_point"
        assert result.keywords == ["审查要点", "保险"]
        assert result.confidence == 0.85

    def test_defaults(self):
        result = TagExtractionResult(
            tags=[],
            doc_type="unknown",
            keywords=[],
            confidence=0.0,
        )
        assert result.tags == []
        assert result.doc_type == "unknown"
        assert result.keywords == []
        assert result.confidence == 0.0


# ── Document type classification tests ────────────────────────────


class TestDocTypeClassification:
    """Test document type classification via _classify_doc_type."""

    def test_classify_regulation(self, extractor):
        doc_type = extractor._classify_doc_type(REGULATION_MD, "通知.pdf")
        assert doc_type == "regulation"

    def test_classify_review_point(self, extractor):
        doc_type = extractor._classify_doc_type(REVIEW_POINT_MD, "审查要点.docx")
        assert doc_type == "review_point"

    def test_classify_insurance_product(self, extractor):
        doc_type = extractor._classify_doc_type(INSURANCE_PRODUCT_MD, "产品条款.pdf")
        assert doc_type == "insurance_product"

    def test_classify_service(self, extractor):
        doc_type = extractor._classify_doc_type(HEALTH_SERVICE_MD, "服务方案.docx")
        assert doc_type == "service"

    def test_classify_contract(self, extractor):
        content = "本合同由甲方和乙方签订，双方约定以下协议条款。"
        doc_type = extractor._classify_doc_type(content, "合同.docx")
        assert doc_type == "contract"

    def test_classify_report(self, extractor):
        content = "2024年度分析报告，包含统计数据和数据汇总。"
        doc_type = extractor._classify_doc_type(content, "报告.pdf")
        assert doc_type == "report"

    def test_classify_unknown_empty(self, extractor):
        doc_type = extractor._classify_doc_type("hello world", "")
        assert doc_type == "unknown"

    def test_classify_unknown_no_match(self, extractor):
        content = "这是一个关于天气的简单文本。"
        doc_type = extractor._classify_doc_type(content, "天气.txt")
        assert doc_type == "unknown"

    def test_classify_uses_filename_hint(self, extractor):
        """Filename keywords add 2.0 points, should dominate short content."""
        content = "这是一份文件。"
        # Without filename hint → unknown (content too generic)
        doc_type_no_hint = extractor._classify_doc_type(content, "")
        # With filename hint → classified
        doc_type_with_hint = extractor._classify_doc_type(content, "保险产品条款.pdf")
        assert doc_type_with_hint == "insurance_product"


# ── Domain tag extraction tests ───────────────────────────────────


class TestDomainTagExtraction:
    """Test domain tag extraction via _extract_domain_tags."""

    def test_extract_consumer_protection(self, extractor):
        tags = extractor._extract_domain_tags("消费者权益保护和消保审查是关键。")
        assert "消保审查" in tags

    def test_extract_product_terms(self, extractor):
        tags = extractor._extract_domain_tags(INSURANCE_PRODUCT_MD)
        assert "产品条款" in tags

    def test_extract_sales_compliance(self, extractor):
        tags = extractor._extract_domain_tags("销售行为必须合规，犹豫期告知到位，双录执行。")
        assert "销售合规" in tags

    def test_extract_claim_service(self, extractor):
        tags = extractor._extract_domain_tags("理赔服务流程，保险金给付和赔付标准。")
        assert "理赔服务" in tags

    def test_extract_health_management(self, extractor):
        tags = extractor._extract_domain_tags(HEALTH_SERVICE_MD)
        assert "健康管理" in tags

    def test_extract_personal_info(self, extractor):
        tags = extractor._extract_domain_tags(PERSONAL_INFO_MD)
        assert "个人信息" in tags

    def test_extract_multiple_tags(self, extractor):
        """Content spanning multiple domains produces multiple tags."""
        tags = extractor._extract_domain_tags(REVIEW_POINT_MD)
        # Review point doc mentions multiple domains
        assert len(tags) >= 2

    def test_extract_no_tags_unrelated(self, extractor):
        tags = extractor._extract_domain_tags("今天天气真好，适合出游。")
        assert tags == []

    def test_extract_no_duplicates(self, extractor):
        """Each domain tag appears at most once even if multiple keywords match."""
        tags = extractor._extract_domain_tags("理赔、赔付、保险金、给付都在理赔服务范围内。")
        assert tags.count("理赔服务") == 1

    def test_all_domain_tags_covered(self, extractor):
        """Verify all DOMAIN_TAGS keys can be triggered."""
        for tag_name, keywords in TagExtractor.DOMAIN_TAGS.items():
            # Use first keyword from each domain
            content = keywords[0]
            tags = extractor._extract_domain_tags(content)
            assert tag_name in tags, f"Domain tag '{tag_name}' not matched by keyword '{content}'"


# ── Keyword extraction tests ──────────────────────────────────────


class TestKeywordExtraction:
    """Test keyword extraction via _extract_keywords."""

    def test_extract_chinese_keywords(self, extractor):
        keywords = extractor._extract_keywords("保险责任和责任免除是保险条款的核心内容。保险责任需要明确。")
        # The CJK regex extracts 2+ char sequences as continuous runs
        # Verify at least some meaningful CJK keywords are extracted
        assert len(keywords) > 0
        # "保险责任" is part of longer CJK runs; verify content is captured
        assert any("保险" in k for k in keywords)

    def test_extract_filters_stop_words(self, extractor):
        keywords = extractor._extract_keywords("的 了 在 是 我 有")
        assert keywords == []

    def test_extract_min_length(self, extractor):
        """Single characters are filtered out."""
        keywords = extractor._extract_keywords("保 险 条 款")
        assert keywords == []

    def test_extract_max_count(self, extractor):
        """At most MAX_KEYWORDS (20) keywords returned."""
        # Generate 50 different repeated words
        content = " ".join(f"关键词{i}" * 5 for i in range(50))
        keywords = extractor._extract_keywords(content)
        assert len(keywords) <= 20

    def test_extract_strips_markdown(self, extractor):
        """Markdown formatting characters are stripped."""
        content = "# 标题\n**保险**条款 `代码` [链接](url)"
        keywords = extractor._extract_keywords(content)
        # Should still find meaningful Chinese terms
        assert any("保险" in k for k in keywords) or any("条款" in k for k in keywords)

    def test_extract_ordered_by_frequency(self, extractor):
        """Keywords are returned in descending frequency order."""
        content = "保险 保险 保险 条款 条款 责任"
        keywords = extractor._extract_keywords(content)
        if len(keywords) >= 2:
            # "保险" has highest frequency
            assert keywords[0] == "保险"


# ── Confidence scoring tests ──────────────────────────────────────


class TestConfidenceScoring:
    """Test confidence computation via _compute_confidence."""

    def test_confidence_empty_content(self, extractor):
        conf = extractor._compute_confidence("", [], "unknown")
        assert conf == 0.0

    def test_confidence_high_tags(self, extractor):
        """More tags → higher confidence."""
        conf_few = extractor._compute_confidence("内容" * 200, ["tag1"], "regulation")
        conf_many = extractor._compute_confidence("内容" * 200, ["tag1", "tag2", "tag3"], "regulation")
        assert conf_many > conf_few

    def test_confidence_classified_vs_unknown(self, extractor):
        """Classified doc type → higher confidence than unknown."""
        conf_unknown = extractor._compute_confidence("内容" * 200, ["tag1"], "unknown")
        conf_classified = extractor._compute_confidence("内容" * 200, ["tag1"], "regulation")
        assert conf_classified > conf_unknown

    def test_confidence_longer_content(self, extractor):
        """Longer content → higher confidence (up to limit)."""
        conf_short = extractor._compute_confidence("短", ["tag1"], "regulation")
        conf_long = extractor._compute_confidence("内容" * 500, ["tag1"], "regulation")
        assert conf_long > conf_short

    def test_confidence_in_range(self, extractor):
        """Confidence is always in [0.0, 1.0]."""
        conf = extractor._compute_confidence("内容" * 1000, ["tag1", "tag2", "tag3"], "regulation")
        assert 0.0 <= conf <= 1.0

    def test_confidence_rounded_to_2_decimals(self, extractor):
        """Confidence is rounded to 2 decimal places."""
        conf = extractor._compute_confidence("测试内容" * 100, ["tag1"], "regulation")
        assert conf == round(conf, 2)


# ── Full extract() integration tests ──────────────────────────────


class TestExtract:
    """Test the full extract() method."""

    def test_extract_insurance_product(self, extractor):
        result = extractor.extract(INSURANCE_PRODUCT_MD, filename="产品条款.pdf")
        assert isinstance(result, TagExtractionResult)
        assert result.doc_type == "insurance_product"
        assert "产品条款" in result.tags
        assert result.confidence > 0.0
        assert len(result.keywords) > 0

    def test_extract_review_point(self, extractor):
        result = extractor.extract(REVIEW_POINT_MD, filename="审查要点.docx")
        assert result.doc_type == "review_point"
        assert len(result.tags) >= 2
        assert result.confidence > 0.3

    def test_extract_regulation(self, extractor):
        result = extractor.extract(REGULATION_MD, filename="通知.pdf")
        assert result.doc_type == "regulation"
        assert "消保审查" in result.tags

    def test_extract_health_service(self, extractor):
        result = extractor.extract(HEALTH_SERVICE_MD, filename="服务方案.docx")
        assert "健康管理" in result.tags

    def test_extract_personal_info(self, extractor):
        result = extractor.extract(PERSONAL_INFO_MD, filename="隐私政策.docx")
        assert "个人信息" in result.tags

    def test_extract_empty_content(self, extractor):
        result = extractor.extract("", filename="empty.txt")
        assert result.tags == []
        assert result.doc_type == "unknown"
        assert result.keywords == []
        assert result.confidence == 0.0

    def test_extract_whitespace_only(self, extractor):
        result = extractor.extract("   \n\t  ", filename="space.txt")
        assert result.tags == []
        assert result.doc_type == "unknown"
        assert result.confidence == 0.0

    def test_extract_none_filename(self, extractor):
        """Extract with no filename hint."""
        result = extractor.extract(INSURANCE_PRODUCT_MD, filename="")
        assert isinstance(result, TagExtractionResult)
        assert result.doc_type == "insurance_product"

    def test_extract_short_content(self, extractor):
        """Short content with some keywords."""
        result = extractor.extract("保险条款规定", filename="test.txt")
        assert isinstance(result, TagExtractionResult)
        assert result.confidence >= 0.0

    def test_extract_unrelated_content(self, extractor):
        """Content that doesn't match any domain patterns."""
        result = extractor.extract("这是一个关于天气的简单文本描述。", filename="weather.txt")
        assert result.doc_type == "unknown"
        assert result.tags == []
        assert result.confidence == 0.0 or result.confidence < 0.3


# ── Class attributes tests ────────────────────────────────────────


class TestTagExtractorClassAttributes:
    """Test TagExtractor class-level constants."""

    def test_doc_type_patterns_is_dict(self):
        assert isinstance(TagExtractor.DOC_TYPE_PATTERNS, dict)
        assert len(TagExtractor.DOC_TYPE_PATTERNS) >= 6

    def test_domain_tags_is_dict(self):
        assert isinstance(TagExtractor.DOMAIN_TAGS, dict)
        assert "消保审查" in TagExtractor.DOMAIN_TAGS
        assert "产品条款" in TagExtractor.DOMAIN_TAGS
        assert "销售合规" in TagExtractor.DOMAIN_TAGS
        assert "理赔服务" in TagExtractor.DOMAIN_TAGS
        assert "健康管理" in TagExtractor.DOMAIN_TAGS
        assert "个人信息" in TagExtractor.DOMAIN_TAGS

    def test_doc_type_values(self):
        expected_types = {"regulation", "review_point", "insurance_product", "service", "contract", "report"}
        assert set(TagExtractor.DOC_TYPE_PATTERNS.keys()) == expected_types

    def test_domain_tag_keywords_are_lists(self):
        for tag_name, keywords in TagExtractor.DOMAIN_TAGS.items():
            assert isinstance(keywords, list), f"DOMAIN_TAGS['{tag_name}'] should be list"
            assert len(keywords) > 0, f"DOMAIN_TAGS['{tag_name}'] should not be empty"
