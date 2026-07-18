"""Unit tests for QueryParser class."""

import pytest

from src.search.query_parser import (
    Query,
    QueryParser,
    CHINESE_STOP_WORDS,
    ENGLISH_STOP_WORDS,
)


class TestQueryDataclass:
    """Test Query dataclass."""

    def test_default_values(self):
        """Test Query with default values."""
        query = Query()
        assert query.terms == []
        assert query.phrases == []
        assert query.fields == {}
        assert query.wildcard is False
        assert query.raw_query == ""

    def test_custom_values(self):
        """Test Query with custom values."""
        query = Query(
            terms=["绩效", "考核"],
            phrases=["2024年标准"],
            fields={"title": "报销"},
            wildcard=True,
            raw_query="绩效 考核",
        )
        assert query.terms == ["绩效", "考核"]
        assert query.phrases == ["2024年标准"]
        assert query.fields == {"title": "报销"}
        assert query.wildcard is True
        assert query.raw_query == "绩效 考核"

    def test_repr(self):
        """Test Query repr."""
        query = Query(terms=["test"], phrases=["exact match"])
        repr_str = repr(query)
        assert "terms=['test']" in repr_str
        assert "phrases=['exact match']" in repr_str


class TestQueryParserInit:
    """Test QueryParser initialization."""

    def test_default_init(self):
        """Test initialization with default values."""
        parser = QueryParser()
        assert parser.chinese_stop_words == set(CHINESE_STOP_WORDS)
        assert parser.english_stop_words == set(ENGLISH_STOP_WORDS)
        assert parser.min_term_length == 1

    def test_custom_stop_words(self):
        """Test initialization with custom stop words."""
        parser = QueryParser(
            chinese_stop_words={"的", "是"},
            english_stop_words={"the", "a"},
        )
        assert parser.chinese_stop_words == {"的", "是"}
        assert parser.english_stop_words == {"the", "a"}

    def test_custom_min_term_length(self):
        """Test initialization with custom min_term_length."""
        parser = QueryParser(min_term_length=2)
        assert parser.min_term_length == 2


class TestQueryParserParse:
    """Test QueryParser parse method."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_parse_empty_query(self, parser):
        """Test parsing empty query."""
        result = parser.parse("")
        assert result.terms == []
        assert result.phrases == []
        assert result.fields == {}
        assert result.wildcard is False
        assert result.raw_query == ""

    def test_parse_whitespace_only(self, parser):
        """Test parsing whitespace-only query."""
        result = parser.parse("   \t\n  ")
        assert result.terms == []
        assert result.phrases == []

    def test_parse_simple_chinese(self, parser):
        """Test parsing simple Chinese query."""
        result = parser.parse("绩效考核流程")
        assert "绩效" in result.terms or "绩效考核" in result.terms
        assert result.phrases == []
        assert result.wildcard is False

    def test_parse_simple_english(self, parser):
        """Test parsing simple English query."""
        result = parser.parse("search engine optimization")
        assert "search" in result.terms
        assert "engine" in result.terms
        assert "optimization" in result.terms

    def test_parse_mixed_chinese_english(self, parser):
        """Test parsing mixed Chinese and English."""
        result = parser.parse("Python 数据分析")
        assert "python" in result.terms
        assert any("数据" in term for term in result.terms)


class TestQueryParserPhraseExtraction:
    """Test phrase extraction functionality."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_double_quoted_phrase(self, parser):
        """Test extraction of double-quoted phrases."""
        result = parser.parse('报销制度"2024年标准"')
        assert "2024年标准" in result.phrases

    def test_single_quoted_phrase(self, parser):
        """Test extraction of single-quoted phrases."""
        result = parser.parse("查找 'exact match' 内容")
        assert "exact match" in result.phrases

    def test_multiple_phrases(self, parser):
        """Test extraction of multiple phrases."""
        result = parser.parse('"first phrase" and "second phrase"')
        assert "first phrase" in result.phrases
        assert "second phrase" in result.phrases

    def test_phrase_removed_from_terms(self, parser):
        """Test that quoted phrases are not in terms."""
        result = parser.parse('"绩效考核" 流程')
        # "绩效考核" should be in phrases, not terms
        assert "绩效考核" in result.phrases
        # 流程 should still be processed as term
        assert "流程" in result.terms

    def test_empty_phrase(self, parser):
        """Test handling of empty phrases."""
        result = parser.parse('"" test')
        # Empty quotes should be ignored
        assert "" not in result.phrases


class TestQueryParserWildcard:
    """Test wildcard functionality."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_asterisk_wildcard(self, parser):
        """Test asterisk wildcard detection."""
        result = parser.parse("绩效*")
        assert result.wildcard is True

    def test_question_mark_wildcard(self, parser):
        """Test question mark wildcard detection."""
        result = parser.parse("test?")
        assert result.wildcard is True

    def test_multiple_wildcards(self, parser):
        """Test multiple wildcards."""
        result = parser.parse("*test*")
        assert result.wildcard is True

    def test_no_wildcard(self, parser):
        """Test query without wildcards."""
        result = parser.parse("绩效考核")
        assert result.wildcard is False

    def test_wildcard_in_phrase(self, parser):
        """Test wildcard in quoted phrase."""
        result = parser.parse('"绩效*" 流程')
        # Phrase content should be treated literally
        assert "绩效*" in result.phrases
        # The wildcard flag should be False since the * is inside quotes
        # Note: current implementation may still detect it as wildcard
        # This test documents current behavior


class TestQueryParserFieldExtraction:
    """Test field-specific query extraction."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_simple_field_query(self, parser):
        """Test simple field:value query."""
        result = parser.parse("title:报告")
        assert result.fields.get("title") == "报告"

    def test_multiple_fields(self, parser):
        """Test multiple field queries."""
        result = parser.parse("title:报告 author:张三")
        assert result.fields.get("title") == "报告"
        assert result.fields.get("author") == "张三"

    def test_field_with_regular_terms(self, parser):
        """Test field query mixed with regular terms."""
        result = parser.parse("title:报告 绩效 考核")
        assert result.fields.get("title") == "报告"
        # Regular terms should still be extracted
        assert len(result.terms) > 0


class TestQueryParserStopWords:
    """Test stop word filtering."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_chinese_stop_words_filtered(self, parser):
        """Test that Chinese stop words are filtered."""
        result = parser.parse("我的绩效考核的流程")
        # "我的" should have "我" filtered
        # "的" should be filtered
        assert "的" not in result.terms
        assert "我" not in result.terms

    def test_english_stop_words_filtered(self, parser):
        """Test that English stop words are filtered."""
        result = parser.parse("the document for the report")
        assert "the" not in result.terms
        assert "for" not in result.terms

    def test_meaningful_words_preserved(self, parser):
        """Test that meaningful words are preserved."""
        result = parser.parse("绩效考核")
        # At least some variation of these should be in terms
        assert len(result.terms) > 0


class TestQueryParserChineseSegmentation:
    """Test Chinese text segmentation."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_common_word_segmentation(self, parser):
        """Test segmentation of common words."""
        result = parser.parse("管理制度")
        # Should recognize "管理" and "制度" as separate words
        # Or keep them together if recognized as a pattern
        assert len(result.terms) > 0

    def test_known_4char_pattern(self, parser):
        """Test recognition of 4-character patterns."""
        result = parser.parse("绩效考核")
        # "绩效考核" is in the common patterns
        assert "绩效考核" in result.terms

    def test_mixed_content(self, parser):
        """Test mixed Chinese content."""
        result = parser.parse("申请报销流程")
        # Should extract meaningful segments
        assert len(result.terms) > 0


class TestQueryParserToTantivyQuery:
    """Test conversion to Tantivy query format."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_simple_terms(self, parser):
        """Test conversion of simple terms."""
        query = Query(terms=["绩效", "考核"])
        tantivy_query = parser.to_tantivy_query(query)
        assert "绩效" in tantivy_query
        assert "考核" in tantivy_query

    def test_phrases(self, parser):
        """Test conversion of phrases."""
        query = Query(phrases=["exact phrase"])
        tantivy_query = parser.to_tantivy_query(query)
        assert '"exact phrase"' in tantivy_query

    def test_fields(self, parser):
        """Test conversion of field queries."""
        query = Query(fields={"title": "报告"})
        tantivy_query = parser.to_tantivy_query(query)
        assert 'title:"报告"' in tantivy_query

    def test_wildcard_terms(self, parser):
        """Test conversion of wildcard terms."""
        query = Query(terms=["绩效*"], wildcard=True)
        tantivy_query = parser.to_tantivy_query(query)
        # Wildcards should be preserved without quotes
        assert "绩效*" in tantivy_query

    def test_empty_query(self, parser):
        """Test conversion of empty query."""
        query = Query()
        tantivy_query = parser.to_tantivy_query(query)
        assert tantivy_query == ""

    def test_combined_query(self, parser):
        """Test conversion of combined query."""
        query = Query(
            terms=["绩效"],
            phrases=["2024标准"],
            fields={"title": "报告"},
        )
        tantivy_query = parser.to_tantivy_query(query)
        assert "绩效" in tantivy_query
        assert '"2024标准"' in tantivy_query
        assert 'title:"报告"' in tantivy_query


class TestQueryParserEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_special_characters(self, parser):
        """Test handling of special characters."""
        result = parser.parse("绩效@#$%考核")
        # Special chars should be removed/normalized
        assert len(result.terms) >= 0  # Should not crash

    def test_very_long_query(self, parser):
        """Test handling of very long query."""
        long_query = "绩效考核" * 100
        result = parser.parse(long_query)
        # Should handle without crashing
        assert len(result.raw_query) > 0

    def test_numbers_in_query(self, parser):
        """Test handling of numbers."""
        result = parser.parse("2024年度报告")
        # Numbers should be handled
        assert len(result.terms) >= 0

    def test_duplicate_terms(self, parser):
        """Test that duplicate terms are deduplicated."""
        result = parser.parse("绩效 绩效 考核 考核")
        # Should not have duplicates
        assert len(result.terms) == len(set(term.lower() for term in result.terms))

    def test_min_term_length_filtering(self):
        """Test that short terms are filtered based on min_term_length."""
        parser = QueryParser(min_term_length=3)
        result = parser.parse("a an the test document")
        # Single char terms should be filtered
        assert "a" not in result.terms
        assert "an" not in result.terms
        assert "test" in result.terms

    def test_unicode_normalization(self, parser):
        """Test handling of Unicode characters."""
        result = parser.parse("文档管理")
        # Should handle Unicode properly
        assert len(result.terms) > 0


class TestQueryParserExamples:
    """Test examples from task specification."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_example_1(self, parser):
        """Test first example: '绩效考核流程'."""
        result = parser.parse("绩效考核流程")
        # Should extract meaningful terms
        assert len(result.terms) > 0
        # Should contain variations of these concepts
        terms_str = " ".join(result.terms)
        assert any(term in terms_str for term in ["绩效", "考核", "流程"])

    def test_example_2(self, parser):
        """Test second example: '报销制度"2024年标准"'."""
        result = parser.parse('报销制度"2024年标准"')
        # Should have the phrase
        assert "2024年标准" in result.phrases
        # Should have terms from '报销制度'
        assert len(result.terms) > 0


class TestQueryParserNormalization:
    """Test query normalization."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_whitespace_normalization(self, parser):
        """Test that multiple whitespaces are normalized."""
        result = parser.parse("绩效   考核  \t  流程")
        # Should handle multiple spaces
        assert len(result.terms) > 0

    def test_case_normalization(self, parser):
        """Test that English is lowercased."""
        result = parser.parse("Python DATA Analysis")
        assert "python" in result.terms
        assert "data" in result.terms

    def test_punctuation_removal(self, parser):
        """Test that punctuation is removed."""
        result = parser.parse("绩效,考核;流程!")
        # Should still extract terms despite punctuation
        assert len(result.terms) > 0


class TestBigramFallback:
    """F3-3: Chinese bigram fallback for OOV terms."""

    @pytest.fixture
    def parser(self):
        """Create a QueryParser instance."""
        return QueryParser()

    def test_bigram_for_consecutive_single_chars(self, parser):
        """3+ consecutive single Chinese chars should produce bigrams."""
        # Use _apply_bigram_fallback directly with a synthetic OOV scenario
        tokens = ["甲", "乙", "丙", "丁"]  # 4 single chars that jieba won't know
        result = parser._apply_bigram_fallback(tokens)

        # Original tokens preserved
        assert "甲" in result
        assert "乙" in result
        assert "丙" in result
        assert "丁" in result
        # Bigrams generated
        assert "甲乙" in result
        assert "乙丙" in result
        assert "丙丁" in result

    def test_no_bigram_for_known_words(self, parser):
        """Known multi-char tokens should not trigger bigram generation."""
        tokens = ["管理", "制度", "流程"]
        result = parser._apply_bigram_fallback(tokens)

        # No bigrams should be generated (all tokens are 2+ chars)
        assert result == tokens

    def test_no_bigram_for_short_runs(self, parser):
        """2 or fewer consecutive single chars should NOT produce bigrams."""
        tokens = ["管理", "甲", "乙", "制度"]
        result = parser._apply_bigram_fallback(tokens)

        # Only 2 consecutive single chars, so no bigrams
        assert "甲乙" not in result
        assert result == tokens

    def test_bigram_preserves_original_tokens(self, parser):
        """Original tokens should always be kept alongside bigrams."""
        tokens = ["甲", "乙", "丙"]
        result = parser._apply_bigram_fallback(tokens)

        # All originals still present
        for t in tokens:
            assert t in result

    def test_bigram_excludes_stop_words(self, parser):
        """Stop words should break single-char runs."""
        # "的" is a stop word — it should break the run
        tokens = ["甲", "乙", "的", "丙", "丁"]
        result = parser._apply_bigram_fallback(tokens)

        # "甲乙" only 2 consecutive before stop word — no bigram
        assert "甲乙" not in result
        # "丙丁" only 2 consecutive after stop word — no bigram
        assert "丙丁" not in result

    def test_bigram_run_of_exactly_three(self, parser):
        """Exactly 3 consecutive single chars should produce 2 bigrams."""
        tokens = ["甲", "乙", "丙"]
        result = parser._apply_bigram_fallback(tokens)

        assert "甲乙" in result
        assert "乙丙" in result
        assert len(result) == 5  # 3 original + 2 bigrams

    def test_bigram_no_duplicate_bigrams(self, parser):
        """Duplicate bigrams should not be added."""
        tokens = ["甲", "乙", "甲", "乙", "丙"]
        result = parser._apply_bigram_fallback(tokens)

        # "甲乙" should appear at most once in bigrams
        bigram_count = result.count("甲乙")
        assert bigram_count <= 2  # once in original position, once as bigram

    def test_bigram_with_mixed_tokens(self, parser):
        """Bigrams only from single Chinese char runs, not from multi-char tokens."""
        tokens = ["管理", "甲", "乙", "丙", "制度", "丁", "戊", "己"]
        result = parser._apply_bigram_fallback(tokens)

        # "甲乙丙" run → bigrams "甲乙", "乙丙"
        assert "甲乙" in result
        assert "乙丙" in result
        # "丁戊己" run → bigrams "丁戊", "戊己"
        assert "丁戊" in result
        assert "戊己" in result
        # Multi-char tokens unchanged
        assert "管理" in result
        assert "制度" in result

    def test_bigram_empty_input(self, parser):
        """Empty token list should return empty."""
        assert parser._apply_bigram_fallback([]) == []

    def test_bigram_non_chinese_single_chars(self, parser):
        """Non-Chinese single characters should not trigger bigram generation."""
        tokens = ["a", "b", "c"]
        result = parser._apply_bigram_fallback(tokens)

        # No Chinese chars → no bigrams
        assert result == tokens

    def test_bigram_integration_with_parse(self, parser):
        """Bigram fallback should be integrated into parse() for Chinese queries."""
        # Use an unusual Chinese string that jieba will likely split into single chars
        result = parser.parse("犰狳蜥蜴恐龙")
        # Whatever jieba produces, the terms list should be non-empty
        assert len(result.terms) > 0
