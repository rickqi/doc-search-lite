"""TagExtractor — Extract document tags from Markdown content using keyword patterns.

Lightweight tag extraction (no LLM calls, zero cost). Uses keyword matching
against insurance/regulatory domain patterns to classify documents and extract
relevant tags for recall-oriented search.

Usage:
    >>> from src.converter.tag_extractor import TagExtractor, TagExtractionResult
    >>> extractor = TagExtractor()
    >>> result = extractor.extract("保险条款规定...", filename="产品条款.pdf")
    >>> print(result.tags)
    ['产品条款', '保险产品']
    >>> print(result.doc_type)
    'insurance_product'
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TagExtractionResult:
    """Result of tag extraction from a document.

    Attributes:
        tags: Extracted domain tags (e.g. ["消保审查", "产品条款"]).
        doc_type: Document type classification (e.g. "regulation", "review_point").
        keywords: Key terms extracted from the content.
        confidence: Extraction confidence in [0.0, 1.0].
    """

    tags: List[str]
    doc_type: str
    keywords: List[str]
    confidence: float


class TagExtractor:
    """Extract document tags from Markdown content using keyword patterns.

    Uses keyword matching against predefined domain tag dictionaries.
    No LLM calls — purely keyword-based, zero API cost.

    The extractor identifies:
    1. Document type (regulation, review_point, insurance_product, service, contract, report)
    2. Domain tags (消保审查, 产品条款, 销售合规, 理赔服务, 健康管理, 个人信息)
    3. Key terms (high-frequency meaningful words)
    4. Confidence score based on match density
    """

    # Document type patterns: doc_type → list of indicative keywords
    DOC_TYPE_PATTERNS: Dict[str, List[str]] = {
        "regulation": [
            "制度", "规定", "办法", "通知", "公告", "条例", "规范",
            "实施细则", "指导意见", "监管规定",
        ],
        "review_point": [
            "审查要点", "检查要点", "审查标准", "要点", "审查项目",
            "检查项目", "审查内容",
        ],
        "insurance_product": [
            "保险", "保费", "保额", "责任范围", "被保险人", "投保",
            "保险产品", "保险责任", "责任免除", "等待期", "免赔额",
        ],
        "service": [
            "服务", "健康管理", "医疗服务", "理赔", "客户服务",
            "增值服务", "就医绿通", "二次诊疗",
        ],
        "contract": [
            "合同", "协议", "条款", "甲方", "乙方", "签约", "约定",
        ],
        "report": [
            "报告", "分析", "统计", "数据", "年报", "季报", "月报",
        ],
    }

    # Insurance domain keywords for tagging: tag_name → list of keywords
    DOMAIN_TAGS: Dict[str, List[str]] = {
        "消保审查": [
            "消费者权益保护", "消保", "投诉", "信息披露", "适当性",
            "消费者保护", "投诉处理", "消保审查",
        ],
        "产品条款": [
            "保险条款", "责任免除", "保险责任", "等待期", "免赔额",
            "产品条款", "保险金额", "保障范围", "保险期间",
        ],
        "销售合规": [
            "销售行为", "犹豫期", "回访", "双录", "提示说明",
            "销售合规", "误导销售", "销售管理", "代理销售",
        ],
        "理赔服务": [
            "理赔", "赔付", "报销", "保险金", "给付",
            "理赔服务", "理赔流程", "理赔时效", "快速理赔",
        ],
        "健康管理": [
            "健康管理", "就医绿通", "二次诊疗", "专家预约",
            "健康服务", "体检", "慢病管理", "健康咨询",
        ],
        "个人信息": [
            "个人信息保护", "数据安全", "隐私", "信息收集",
            "个人信息", "数据保护", "信息安全", "隐私政策",
        ],
    }

    # Chinese stop words for keyword extraction
    _STOP_WORDS = frozenset({
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
        "它", "们", "那", "些", "什么", "吗", "吧", "啊", "呢", "把",
        "被", "让", "给", "对", "为", "与", "从", "但", "而", "或",
        "中", "等", "可以", "可", "该", "其", "此", "以", "及", "之",
        "于", "按", "照", "根据", "通过", "按照",
        # Common markdown noise
        "nbsp", "amp", "lt", "gt", "mdash", "ndash",
    })

    # Minimum keyword length (characters)
    _MIN_KEYWORD_LEN = 2

    # Maximum keywords to return
    _MAX_KEYWORDS = 20

    def extract(self, markdown: str, filename: str = "") -> TagExtractionResult:
        """Extract tags from Markdown content.

        Args:
            markdown: Converted Markdown content.
            filename: Original filename (used as hint for classification).

        Returns:
            TagExtractionResult with tags, doc_type, keywords, and confidence.
        """
        if not markdown or not markdown.strip():
            return TagExtractionResult(
                tags=[],
                doc_type="unknown",
                keywords=[],
                confidence=0.0,
            )

        # 1. Classify document type
        doc_type = self._classify_doc_type(markdown, filename)

        # 2. Extract domain tags based on keyword matching
        tags = self._extract_domain_tags(markdown)

        # 3. Extract key terms (high-frequency meaningful words)
        keywords = self._extract_keywords(markdown)

        # 4. Compute confidence based on match density
        confidence = self._compute_confidence(markdown, tags, doc_type)

        logger.debug(
            "TagExtractor: filename=%r, doc_type=%s, tags=%s, confidence=%.2f",
            filename, doc_type, tags, confidence,
        )

        return TagExtractionResult(
            tags=tags,
            doc_type=doc_type,
            keywords=keywords,
            confidence=confidence,
        )

    def _classify_doc_type(self, markdown: str, filename: str) -> str:
        """Classify the document type based on content and filename.

        Scoring: each pattern keyword found in content adds 1 point.
        Filename hints add 2 points. Type with highest score wins.

        Args:
            markdown: Markdown content.
            filename: Original filename.

        Returns:
            Document type string (e.g. "regulation", "review_point").
        """
        content_lower = markdown.lower()
        filename_lower = filename.lower()

        scores: Dict[str, float] = {}
        for doc_type, patterns in self.DOC_TYPE_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if pattern in content_lower:
                    score += 1.0
                if pattern in filename_lower:
                    score += 2.0
            scores[doc_type] = score

        if not scores or max(scores.values()) == 0:
            return "unknown"

        return max(scores, key=lambda k: scores[k])

    def _extract_domain_tags(self, markdown: str) -> List[str]:
        """Extract domain tags based on keyword matching.

        A domain tag is included if at least one of its keywords appears
        in the content.

        Args:
            markdown: Markdown content.

        Returns:
            List of matched domain tag names.
        """
        matched_tags = []
        for tag_name, keywords in self.DOMAIN_TAGS.items():
            for keyword in keywords:
                if keyword in markdown:
                    matched_tags.append(tag_name)
                    break  # One match is enough per tag

        return matched_tags

    def _extract_keywords(self, markdown: str) -> List[str]:
        """Extract key terms from content.

        Uses simple regex to find Chinese character sequences (2+ chars)
        and filters out stop words. Returns top keywords by frequency.

        Args:
            markdown: Markdown content.

        Returns:
            List of key terms, ordered by frequency (descending).
        """
        # Strip markdown formatting characters
        cleaned = re.sub(r"[#*`>\-|\[\](){},.!?;:'\"/\\]", " ", markdown)

        # Find Chinese word sequences (2+ consecutive CJK characters)
        cjk_pattern = re.compile(r"[\u4e00-\u9fff]{2,}")
        words = cjk_pattern.findall(cleaned)

        # Count frequencies, excluding stop words
        freq: Dict[str, int] = {}
        for word in words:
            if word in self._STOP_WORDS:
                continue
            if len(word) < self._MIN_KEYWORD_LEN:
                continue
            freq[word] = freq.get(word, 0) + 1

        # Sort by frequency descending, return top keywords
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[: self._MAX_KEYWORDS]]

    def _compute_confidence(
        self,
        markdown: str,
        tags: List[str],
        doc_type: str,
    ) -> float:
        """Compute extraction confidence based on match density.

        Confidence factors:
        - Number of matched domain tags (more = higher confidence)
        - Document type certainty (not "unknown" = higher confidence)
        - Content length (longer content = more reliable extraction)

        Args:
            markdown: Original Markdown content.
            tags: Extracted domain tags.
            doc_type: Classified document type.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        if not markdown.strip():
            return 0.0

        # Base confidence from tag count (max 0.5 from 3+ tags)
        tag_factor = min(len(tags) / 3.0, 1.0) * 0.5

        # Document type certainty (0.2 if classified, 0.0 if unknown)
        type_factor = 0.2 if doc_type != "unknown" else 0.0

        # Content length factor (short content = less reliable)
        content_len = len(markdown.strip())
        length_factor = min(content_len / 500.0, 1.0) * 0.3

        confidence = tag_factor + type_factor + length_factor
        return round(min(max(confidence, 0.0), 1.0), 2)
