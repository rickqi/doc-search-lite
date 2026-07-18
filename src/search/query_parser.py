"""
Query parser for converting natural language queries to Tantivy queries.

Supports:
- Keyword extraction
- Phrase queries (quoted text)
- Wildcards (* and ?)
- Chinese text segmentation (using jieba when available, with rule-based fallback)
- Stop word filtering
"""

import re
from dataclasses import dataclass, field


@dataclass
class Query:
    """
    Structured query object for search operations.

    Attributes:
        terms: List of search terms extracted from the query
        phrases: List of exact phrases (from quoted text)
        fields: Dictionary of field-specific queries
        wildcard: Boolean indicating if wildcards are used
        raw_query: Original raw query string
    """

    terms: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    wildcard: bool = False
    raw_query: str = ""

    def __repr__(self) -> str:
        return (
            f"Query(terms={self.terms}, phrases={self.phrases}, "
            f"fields={self.fields}, wildcard={self.wildcard})"
        )


# Chinese common stop words
CHINESE_STOP_WORDS = frozenset(
    [
        # Particles
        "的",
        "了",
        "和",
        "与",
        "或",
        "及",
        "等",
        "之",
        "所",
        "而",
        # Pronouns
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "自己",
        "这",
        "那",
        # Prepositions
        "在",
        "从",
        "向",
        "到",
        "对",
        "为",
        "把",
        "被",
        "让",
        "给",
        # Conjunctions
        "因为",
        "所以",
        "但是",
        "如果",
        "虽然",
        "并且",
        "而且",
        "或者",
        "以及",
        # Adverbs
        "很",
        "也",
        "都",
        "就",
        "又",
        "才",
        "只",
        "还",
        "已",
        "再",
        # Auxiliary verbs
        "是",
        "有",
        "会",
        "能",
        "可以",
        "应该",
        "要",
        "想",
        "需",
        # Question words
        "什么",
        "怎么",
        "如何",
        "为什么",
        "哪",
        "哪里",
        "谁",
        "多少",
        # Common measure words
        "个",
        "只",
        "条",
        "件",
        "种",
        "样",
        "次",
        "些",
        # Time expressions
        "时",
        "后",
        "前",
        "当",
        "现在",
        "以后",
        "之前",
        # Negations
        "不",
        "没",
        "无",
        "非",
        "未",
        # Other common words
        "这",
        "那",
        "此",
        "彼",
        "该",
        "其",
        "上",
        "下",
        "中",
        "内",
        "外",
        "将",
        "得",
        "着",
        "过",
        "啊",
        "吗",
        "呢",
        "吧",
        "呀",
        "哦",
    ]
)

# English common stop words (for mixed queries)
ENGLISH_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "would",
        "could",
        "ought",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
    ]
)


class QueryParser:
    """
    Parser for converting natural language queries to structured Query objects.

    This parser handles:
    - Extraction of quoted phrases
    - Wildcard pattern detection (* and ?)
    - Chinese text segmentation (using jieba when available, with rule-based fallback)
    - Stop word filtering
    - Field-specific queries (field:value syntax)
    """

    # Pattern for quoted phrases (supports both " and ')
    PHRASE_PATTERN = re.compile(r'["\']([^"\']+)["\']')

    # Pattern for field queries (field:value)
    FIELD_PATTERN = re.compile(r"(\w+):([^\s]+)")

    # Pattern for wildcards
    WILDCARD_PATTERN = re.compile(r"[*?]")

    # Pattern for punctuation and special characters (except wildcards)
    PUNCTUATION_PATTERN = re.compile(r"[^\w\u4e00-\u9fff\s*?]+")

    # Pattern for whitespace normalization
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        chinese_stop_words: set[str] | None = None,
        english_stop_words: set[str] | None = None,
        min_term_length: int = 1,
    ):
        """
        Initialize the QueryParser.

        Args:
            chinese_stop_words: Custom set of Chinese stop words (uses default if None)
            english_stop_words: Custom set of English stop words (uses default if None)
            min_term_length: Minimum length for a term to be included
        """
        self.chinese_stop_words = chinese_stop_words or set(CHINESE_STOP_WORDS)
        self.english_stop_words = english_stop_words or set(ENGLISH_STOP_WORDS)
        self.min_term_length = min_term_length

        # Check if jieba is available
        self._jieba_available = False
        try:
            import jieba
            self._jieba_available = True
        except ImportError:
            pass

    def parse(self, query: str) -> Query:
        """
        Parse a natural language query into a structured Query object.

        Args:
            query: The raw query string

        Returns:
            Query object with extracted terms, phrases, fields, and wildcard flag
        """
        result = Query(raw_query=query)

        if not query or not query.strip():
            return result

        # Extract and remove quoted phrases
        phrases, remaining = self._extract_phrases(query)
        result.phrases = phrases

        # Extract field-specific queries
        fields, remaining = self._extract_fields(remaining)
        result.fields = fields

        # Check for wildcards
        result.wildcard = bool(self.WILDCARD_PATTERN.search(remaining))

        # Normalize whitespace and punctuation
        cleaned = self._normalize(remaining)

        # Segment and extract terms
        terms = self._extract_terms(cleaned)

        # Filter stop words and short terms
        result.terms = self._filter_terms(terms)

        return result

    def _extract_phrases(self, query: str) -> tuple[list[str], str]:
        """
        Extract quoted phrases from the query.

        Args:
            query: The raw query string

        Returns:
            Tuple of (list of phrases, remaining query string)
        """
        phrases = self.PHRASE_PATTERN.findall(query)
        remaining = self.PHRASE_PATTERN.sub("", query)
        return phrases, remaining

    def _extract_fields(self, query: str) -> tuple[dict[str, str], str]:
        """
        Extract field-specific queries (field:value syntax).

        Args:
            query: The query string

        Returns:
            Tuple of (field dict, remaining query string)
        """
        fields = {}
        matches = list(self.FIELD_PATTERN.finditer(query))

        for match in matches:
            field_name = match.group(1)
            field_value = match.group(2)
            fields[field_name] = field_value

        remaining = self.FIELD_PATTERN.sub("", query)
        return fields, remaining

    def _normalize(self, query: str) -> str:
        """
        Normalize the query by removing punctuation and collapsing whitespace.

        Args:
            query: The query string

        Returns:
            Normalized query string
        """
        # Remove punctuation (keep wildcards)
        cleaned = self.PUNCTUATION_PATTERN.sub(" ", query)
        # Normalize whitespace
        cleaned = self.WHITESPACE_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    def _extract_terms(self, query: str) -> list[str]:
        """
        Extract and segment terms from the query.

        For Chinese text, uses jieba segmentation when available (with rule-based fallback).
        For mixed text, handles both Chinese and English.

        Args:
            query: The normalized query string

        Returns:
            List of extracted terms
        """
        if not query:
            return []

        terms = []

        # Process each token
        for token in query.split():
            if not token:
                continue

            # Check if token contains Chinese characters
            if self._contains_chinese(token):
                # Segment Chinese text
                terms.extend(self._segment_chinese(token))
            else:
                # English or other text - use as is
                terms.append(token.lower())

        return terms

    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _segment_chinese_fallback(self, text: str) -> list[str]:
        """
        Fallback rule-based Chinese text segmentation when jieba is not available.

        Uses a combination of:
        1. Common word boundary detection
        2. Bigram/trigram extraction for unknown patterns
        3. Single character fallback

        Args:
            text: Chinese text to segment

        Returns:
            List of segmented terms
        """
        if not text:
            return []

        terms = []

        # Common Chinese word patterns (2-4 characters)
        common_patterns = [
            # Management/Business
            "管理",
            "制度",
            "流程",
            "规定",
            "办法",
            "条例",
            "政策",
            "标准",
            "规范",
            "考核",
            "绩效",
            "目标",
            "计划",
            "总结",
            "报告",
            "审批",
            "执行",
            "监督",
            "考核",
            "评估",
            "分析",
            "研究",
            "开发",
            "设计",
            "实施",
            "推广",
            "运营",
            # HR
            "招聘",
            "培训",
            "薪资",
            "福利",
            "休假",
            "请假",
            "加班",
            "出差",
            "报销",
            "人事",
            "员工",
            "部门",
            "岗位",
            "职位",
            "职责",
            "任职",
            "晋升",
            "调岗",
            # Finance
            "财务",
            "预算",
            "成本",
            "费用",
            "资金",
            "收入",
            "支出",
            "利润",
            "税务",
            # Operations
            "采购",
            "销售",
            "合同",
            "协议",
            "项目",
            "任务",
            "进度",
            "质量",
            "风险",
            # Common verbs
            "申请",
            "审批",
            "确认",
            "核实",
            "处理",
            "解决",
            "协调",
            "沟通",
            "反馈",
            "提交",
            "修改",
            "更新",
            "删除",
            "添加",
            "设置",
            "配置",
            "维护",
            "优化",
            # Common nouns
            "文件",
            "资料",
            "信息",
            "数据",
            "系统",
            "平台",
            "工具",
            "资源",
            "服务",
            "产品",
            "服务",
            "客户",
            "用户",
            "公司",
            "企业",
            "组织",
            "团队",
            "成员",
            # Time
            "时间",
            "日期",
            "期限",
            "周期",
            "年度",
            "季度",
            "月度",
            "周报",
            "日报",
            # Documents
            "通知",
            "公告",
            "说明",
            "指南",
            "手册",
            "表格",
            "模板",
            "清单",
            "目录",
        ]

        # Add 4-character common words
        common_4char = [
            "绩效考核",
            "绩效评估",
            "薪酬管理",
            "人力资源",
            "项目管理",
            "质量管理",
            "风险管理",
            "财务管理",
            "成本控制",
            "预算管理",
            "工作流程",
            "操作规程",
            "申请流程",
            "审批流程",
            "招聘流程",
            "入职流程",
            "离职流程",
            "培训计划",
            "报销制度",
            "考勤制度",
            "休假制度",
            "加班制度",
            "薪酬制度",
            "晋升制度",
            "安全规范",
            "质量标准",
            "服务标准",
            "管理规范",
            "技术规范",
            "行为准则",
            "年度报告",
            "季度总结",
            "项目计划",
            "工作总结",
            "会议纪要",
            "决策记录",
            "数据分析",
            "问题解决",
            "流程优化",
            "系统配置",
            "资源分配",
            "任务分配",
        ]

        i = 0
        n = len(text)

        while i < n:
            matched = False

            # Try to match 4-character words first
            if i + 4 <= n:
                four_char = text[i : i + 4]
                if four_char in common_4char:
                    terms.append(four_char)
                    i += 4
                    matched = True
                    continue

            # Try to match 2-3 character common patterns
            for length in [3, 2]:
                if i + length <= n:
                    word = text[i : i + length]
                    if word in common_patterns:
                        terms.append(word)
                        i += length
                        matched = True
                        break

            if matched:
                continue

            # Fallback: use bigrams for better matching
            if i + 2 <= n:
                bigram = text[i : i + 2]
                # Check if bigram is meaningful (not just random chars)
                if not self._is_stop_char(text[i]) and not self._is_stop_char(
                    text[i + 1]
                ):
                    terms.append(bigram)
                i += 2
            else:
                # Single character
                if not self._is_stop_char(text[i]):
                    terms.append(text[i])
                i += 1

        return terms

    def _segment_chinese(self, text: str) -> list[str]:
        """
        Chinese text segmentation using jieba or fallback method.

        When jieba is available, applies bigram fallback for OOV terms
        (single-character tokens that may be parts of unknown compound words).

        Args:
            text: Chinese text to segment

        Returns:
            List of segmented terms
        """
        if not text:
            return []

        if self._jieba_available:
            import jieba
            tokens = list(jieba.cut(text))
            tokens = [t for t in tokens if t.strip()]
            return self._apply_bigram_fallback(tokens)
        else:
            return self._segment_chinese_fallback(text)

    def _apply_bigram_fallback(self, tokens: list[str]) -> list[str]:
        """Generate bigram fallbacks for single-character OOV terms.

        When jieba produces consecutive single characters that look like
        they should be a compound term, generate bigrams as additional terms.
        Only triggers for runs of 3+ consecutive single Chinese characters
        (excluding stop words), which are likely OOV compounds.

        Args:
            tokens: List of tokens from jieba segmentation.

        Returns:
            Original tokens plus any generated bigrams (appended, not interleaved).
        """
        if not tokens:
            return tokens

        result = list(tokens)  # Keep original tokens

        # Find runs of consecutive single Chinese characters (excluding stop words)
        i = 0
        while i < len(tokens):
            if (
                len(tokens[i]) == 1
                and self._contains_chinese(tokens[i])
                and tokens[i] not in self.chinese_stop_words
            ):
                # Start of a potential single-char run
                run_start = i
                while (
                    i < len(tokens)
                    and len(tokens[i]) == 1
                    and self._contains_chinese(tokens[i])
                    and tokens[i] not in self.chinese_stop_words
                ):
                    i += 1
                run_end = i

                # If we have 3+ consecutive single chars, generate bigrams
                if run_end - run_start >= 3:
                    run_tokens = tokens[run_start:run_end]
                    for j in range(len(run_tokens) - 1):
                        bigram = run_tokens[j] + run_tokens[j + 1]
                        if bigram not in result:
                            result.append(bigram)
            else:
                i += 1

        return result

    def _is_stop_char(self, char: str) -> bool:
        """Check if a character is a stop word."""
        return char in self.chinese_stop_words

    def _filter_terms(self, terms: list[str]) -> list[str]:
        """
        Filter out stop words and short terms.

        Args:
            terms: List of terms to filter

        Returns:
            Filtered list of terms
        """
        filtered = []
        seen = set()

        for term in terms:
            # Skip if too short
            if len(term) < self.min_term_length:
                continue

            # Skip if it's a Chinese stop word
            if term in self.chinese_stop_words:
                continue

            # Skip if it's an English stop word
            if term.lower() in self.english_stop_words:
                continue

            # Skip duplicates
            term_lower = term.lower()
            if term_lower in seen:
                continue

            seen.add(term_lower)
            filtered.append(term)

        return filtered

    def to_tantivy_query(self, query: Query) -> str:
        """
        Convert a Query object to a Tantivy query string.

        Args:
            query: The Query object to convert

        Returns:
            Tantivy-compatible query string
        """
        parts = []

        # Add field-specific queries
        for field_name, field_value in query.fields.items():
            if self.WILDCARD_PATTERN.search(field_value):
                parts.append(f"{field_name}:{field_value}")
            else:
                parts.append(f'{field_name}:"{field_value}"')

        # Add phrases as exact matches
        for phrase in query.phrases:
            parts.append(f'"{phrase}"')

        # Add terms
        for term in query.terms:
            if self.WILDCARD_PATTERN.search(term):
                parts.append(term)
            else:
                parts.append(term)

        return " ".join(parts) if parts else ""
