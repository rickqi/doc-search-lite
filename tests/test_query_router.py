"""Tests for QueryRouter — keyword-based query-to-index routing."""

import pytest

from src.search.query_router import IndexMeta, QueryRouter


def _hr_meta(path="raw/hr/index"):
    return IndexMeta(
        path=path,
        tags=["hr", "人事"],
        name="hr-docs",
        source_dir="hr",
    )


def _finance_meta(path="raw/finance/index"):
    return IndexMeta(
        path=path,
        tags=["finance", "财务"],
        name="finance-docs",
        source_dir="finance",
    )


def _tech_meta(path="raw/tech/index"):
    return IndexMeta(
        path=path,
        tags=["tech", "技术"],
        name="tech-docs",
        source_dir="tech",
    )


def _legal_meta(path="raw/legal/index"):
    return IndexMeta(
        path=path,
        tags=["legal"],
        name="legal-docs",
        source_dir="legal",
    )


def _general_meta(path="raw/general/index"):
    return IndexMeta(
        path=path,
        tags=[],
        name="general",
    )


# ── IndexMeta tests ──────────────────────────────────────────────


class TestIndexMeta:
    """Test IndexMeta dataclass defaults."""

    def test_defaults(self):
        meta = IndexMeta(path="some/path")
        assert meta.tags == []
        assert meta.profile == "general"
        assert meta.name == ""
        assert meta.doc_count == 0
        assert meta.source_dir == ""

    def test_custom_values(self):
        meta = IndexMeta(
            path="raw/hr/index",
            tags=["hr"],
            profile="legal",
            name="hr-docs",
            doc_count=42,
            source_dir="hr",
        )
        assert meta.tags == ["hr"]
        assert meta.profile == "legal"
        assert meta.doc_count == 42


# ── QueryRouter.route() tests ────────────────────────────────────


class TestQueryRouterRoute:
    """Test QueryRouter.route() main routing logic."""

    def test_route_by_tag_exact_match(self):
        """Tag 'hr' directly in query → hr index matched."""
        router = QueryRouter({"raw/hr/index": _hr_meta()})
        result = router.route("查询hr相关政策")
        assert result == ["raw/hr/index"]

    def test_route_by_tag_chinese(self):
        """Chinese tag '人事' in query → match."""
        router = QueryRouter({"raw/hr/index": _hr_meta()})
        result = router.route("人事部门在哪里")
        assert result == ["raw/hr/index"]

    def test_route_by_domain_keywords(self):
        """Domain keywords like '年假' trigger hr domain when index tagged 'hr'."""
        router = QueryRouter({
            "raw/hr/index": _hr_meta(),
            "raw/finance/index": _finance_meta(),
        })
        result = router.route("年假如何申请")
        assert "raw/hr/index" in result
        # finance should not match 年假
        assert "raw/finance/index" not in result

    def test_route_no_match_returns_all(self):
        """When no index matches, return all indexes (fallback)."""
        router = QueryRouter({
            "raw/hr/index": _hr_meta(),
            "raw/finance/index": _finance_meta(),
        })
        result = router.route("天气怎么样")
        # Fallback: all indexes returned
        assert len(result) == 2

    def test_route_top_k_limit(self):
        """top_k limits number of returned index paths."""
        router = QueryRouter({
            "raw/hr/index": _hr_meta(),
            "raw/finance/index": _finance_meta(),
            "raw/tech/index": _tech_meta(),
        })
        result = router.route("年假如何申请", top_k=1)
        assert len(result) <= 1

    def test_route_empty_indexes(self):
        """Empty index dict → empty result."""
        router = QueryRouter({})
        result = router.route("年假如何申请")
        assert result == []

    def test_route_empty_query(self):
        """Empty query → all indexes returned."""
        router = QueryRouter({
            "raw/hr/index": _hr_meta(),
            "raw/finance/index": _finance_meta(),
        })
        result = router.route("")
        assert len(result) == 2

    def test_route_whitespace_query(self):
        """Whitespace-only query → all indexes returned."""
        router = QueryRouter({"raw/hr/index": _hr_meta()})
        result = router.route("   ")
        assert result == ["raw/hr/index"]

    def test_route_multiple_matches_ranked(self):
        """Multiple matching indexes ranked by score (highest first)."""
        # Build indexes: hr with high relevance, finance with partial relevance
        hr_meta = IndexMeta(path="raw/hr/index", tags=["hr", "年假"])
        fin_meta = IndexMeta(path="raw/finance/index", tags=["hr", "财务"])
        router = QueryRouter({
            "raw/hr/index": hr_meta,
            "raw/finance/index": fin_meta,
        })
        result = router.route("年假")
        # Both match "hr" tag, but hr_meta also has "年假" tag → higher score
        assert result[0] == "raw/hr/index"

    def test_route_by_name(self):
        """Index name matching query."""
        router = QueryRouter({
            "raw/general/index": _general_meta(),
        })
        result = router.route("在general中搜索")
        assert "raw/general/index" in result

    def test_route_by_source_dir(self):
        """Source dir matching query."""
        meta = IndexMeta(path="raw/data/index", tags=[], source_dir="mydocs")
        router = QueryRouter({"raw/data/index": meta})
        result = router.route("mydocs里面的内容")
        assert "raw/data/index" in result

    def test_route_single_index(self):
        """Single index always returned (either matched or fallback)."""
        router = QueryRouter({"raw/hr/index": _hr_meta()})
        result = router.route("随便搜搜")
        assert result == ["raw/hr/index"]

    def test_route_case_insensitive(self):
        """Tag matching is case-insensitive."""
        meta = IndexMeta(path="raw/idx", tags=["HR"])
        router = QueryRouter({"raw/idx": meta})
        result = router.route("hr相关")
        assert "raw/idx" in result


# ── QueryRouter._score_index() tests ────────────────────────────


class TestQueryRouterScoring:
    """Test _score_index scoring logic in detail."""

    def test_score_zero_for_unrelated(self):
        """Unrelated query gets score 0."""
        router = QueryRouter({"raw/hr/index": _hr_meta()})
        score = router._score_index("天气怎么样", _hr_meta())
        assert score == 0.0

    def test_score_tag_exact(self):
        """Exact tag match scores 2.0."""
        router = QueryRouter({"raw/hr/index": _hr_meta()})
        score = router._score_index("hr政策", _hr_meta())
        assert score >= 2.0

    def test_score_domain_keyword_match(self):
        """Domain keyword + tag match scores additional 1.5."""
        router = QueryRouter({"raw/hr/index": _hr_meta()})
        score = router._score_index("年假如何申请", _hr_meta())
        # '年假' is in hr domain keywords, and index is tagged 'hr'
        assert score >= 1.5

    def test_score_name_match(self):
        """Name match adds 1.0."""
        meta = IndexMeta(path="p", tags=[], name="special-docs")
        router = QueryRouter({"p": meta})
        score = router._score_index("special-docs", meta)
        assert score >= 1.0

    def test_score_source_dir_match(self):
        """Source dir match adds 0.5."""
        meta = IndexMeta(path="p", tags=[], source_dir="mydata")
        router = QueryRouter({"p": meta})
        score = router._score_index("mydata", meta)
        assert score >= 0.5

    def test_score_multiple_signals(self):
        """Multiple matching signals accumulate."""
        meta = IndexMeta(path="p", tags=["hr"], name="hr", source_dir="hr")
        router = QueryRouter({"p": meta})
        score = router._score_index("hr相关年假政策", meta)
        # tag match (2.0) + domain match (1.5) + name match (1.0) + source_dir (0.5)
        assert score >= 2.0  # At minimum tag match


# ── QueryRouter.properties tests ────────────────────────────────


class TestQueryRouterProperties:
    """Test QueryRouter properties."""

    def test_indexes_property(self):
        """indexes property returns registered indexes."""
        meta = _hr_meta()
        router = QueryRouter({"raw/hr/index": meta})
        assert "raw/hr/index" in router.indexes
        assert router.indexes["raw/hr/index"] is meta

    def test_domain_keywords_is_class_attribute(self):
        """DOMAIN_KEYWORDS is a class-level constant."""
        assert isinstance(QueryRouter.DOMAIN_KEYWORDS, dict)
        assert "hr" in QueryRouter.DOMAIN_KEYWORDS
        assert "finance" in QueryRouter.DOMAIN_KEYWORDS
        assert "tech" in QueryRouter.DOMAIN_KEYWORDS
        assert "legal" in QueryRouter.DOMAIN_KEYWORDS
        assert "admin" in QueryRouter.DOMAIN_KEYWORDS
        assert "insurance" in QueryRouter.DOMAIN_KEYWORDS


# ── QueryRouter.rewrite_for_routing() tests ──────────────────────


class TestQueryRouterRewrite:
    """Test rewrite_for_routing() method."""

    def test_rewrite_no_match(self):
        """Query without colloquial phrases is unchanged."""
        router = QueryRouter({})
        result = router.rewrite_for_routing("年假如何申请")
        assert result == "年假如何申请"

    def test_rewrite_colloquial_expansion(self):
        """Colloquial phrase triggers expansion."""
        router = QueryRouter({})
        result = router.rewrite_for_routing("报销多少钱")
        assert "费用" in result
        assert "报销多少钱" in result  # original preserved

    def test_rewrite_multiple_matches(self):
        """Multiple colloquial phrases all expanded."""
        router = QueryRouter({})
        result = router.rewrite_for_routing("怎么办手续需要多久")
        assert "流程" in result
        assert "期限" in result

    def test_rewrite_improves_routing(self):
        """Rewriting helps route colloquial queries correctly."""
        router = QueryRouter({
            "raw/finance/index": _finance_meta(),
            "raw/hr/index": _hr_meta(),
        })
        # Without rewrite: "多少钱" doesn't match finance keywords
        # With rewrite: expands to "费用 价格" which matches finance domain
        result = router.route("出差多少钱")
        assert "raw/finance/index" in result

    def test_rewrite_preserves_original(self):
        """Original query text is always preserved in output."""
        router = QueryRouter({})
        result = router.rewrite_for_routing("能不能报销")
        assert result.startswith("能不能报销")

    def test_rewrite_class_attribute_exists(self):
        """_ROUTING_REWRITE_MAP is a class attribute."""
        assert isinstance(QueryRouter._ROUTING_REWRITE_MAP, dict)
        assert "多少钱" in QueryRouter._ROUTING_REWRITE_MAP
        assert "怎么办手续" in QueryRouter._ROUTING_REWRITE_MAP


# ── QueryRouter insurance domain tests ───────────────────────────


def _insurance_meta(path="raw/insurance/index"):
    return IndexMeta(
        path=path,
        tags=["insurance", "保险"],
        name="insurance-docs",
        source_dir="insurance",
    )


class TestQueryRouterInsuranceDomain:
    """Test routing with new insurance domain keywords."""

    def test_insurance_claim_routing(self):
        """Insurance claim keywords route to insurance index."""
        meta = IndexMeta(path="raw/insurance/index", tags=["insurance", "保险"])
        router = QueryRouter({"raw/insurance/index": meta})
        result = router.route("理赔申请流程")
        assert "raw/insurance/index" in result

    def test_insurance_compliance_routing(self):
        """消保审查 keywords route to insurance index."""
        meta = IndexMeta(path="raw/insurance/index", tags=["insurance", "保险"])
        router = QueryRouter({"raw/insurance/index": meta})
        result = router.route("宣传材料消保审查")
        assert "raw/insurance/index" in result

    def test_insurance_product_routing(self):
        """Insurance product keywords route to insurance index."""
        meta = IndexMeta(path="raw/insurance/index", tags=["insurance", "保险"])
        router = QueryRouter({"raw/insurance/index": meta})
        result = router.route("保单续保流程")
        assert "raw/insurance/index" in result

    def test_insurance_vs_hr_routing(self):
        """Insurance queries don't route to hr index."""
        router = QueryRouter({
            "raw/insurance/index": _insurance_meta(),
            "raw/hr/index": _hr_meta(),
        })
        result = router.route("理赔申请")
        assert "raw/insurance/index" in result

    def test_insurance_sales_routing(self):
        """Insurance sales keywords route to insurance index."""
        meta = IndexMeta(path="raw/insurance/index", tags=["insurance", "保险"])
        router = QueryRouter({"raw/insurance/index": meta})
        result = router.route("代理人营销规范")
        assert "raw/insurance/index" in result


# ── QueryRouter.route_by_tags() tests ────────────────────────────


class TestQueryRouterRouteByTags:
    """Test QueryRouter.route_by_tags() for recall scenario."""

    def test_route_by_tags_exact_match(self):
        """Exact tag match routes to correct index."""
        meta = IndexMeta(path="raw/insurance/index", tags=["消保审查", "产品条款"])
        router = QueryRouter({"raw/insurance/index": meta})
        result = router.route_by_tags(["消保审查"])
        assert result == ["raw/insurance/index"]

    def test_route_by_tags_multiple_matches_ranked(self):
        """Indexes with more tag matches rank higher."""
        meta_a = IndexMeta(path="raw/a/index", tags=["消保审查"])
        meta_b = IndexMeta(path="raw/b/index", tags=["消保审查", "产品条款"])
        router = QueryRouter({
            "raw/a/index": meta_a,
            "raw/b/index": meta_b,
        })
        result = router.route_by_tags(["消保审查", "产品条款"])
        # meta_b matches both tags → higher score → first
        assert result[0] == "raw/b/index"

    def test_route_by_tags_no_match_returns_all(self):
        """When no tags match, return all indexes (fallback)."""
        meta_a = IndexMeta(path="raw/a/index", tags=["hr"])
        meta_b = IndexMeta(path="raw/b/index", tags=["finance"])
        router = QueryRouter({"raw/a/index": meta_a, "raw/b/index": meta_b})
        result = router.route_by_tags(["消保审查"])
        # No match → fallback to all
        assert len(result) == 2

    def test_route_by_tags_empty_tags(self):
        """Empty tag list returns all indexes."""
        meta = IndexMeta(path="raw/a/index", tags=["hr"])
        router = QueryRouter({"raw/a/index": meta})
        result = router.route_by_tags([])
        assert result == ["raw/a/index"]

    def test_route_by_tags_empty_indexes(self):
        """Empty index dict returns empty list."""
        router = QueryRouter({})
        result = router.route_by_tags(["消保审查"])
        assert result == []

    def test_route_by_tags_case_insensitive(self):
        """Tag matching is case-insensitive."""
        meta = IndexMeta(path="raw/idx", tags=["HR"])
        router = QueryRouter({"raw/idx": meta})
        result = router.route_by_tags(["hr"])
        assert "raw/idx" in result

    def test_route_by_tags_top_k(self):
        """top_k limits the number of returned indexes."""
        meta_a = IndexMeta(path="raw/a/index", tags=["消保审查"])
        meta_b = IndexMeta(path="raw/b/index", tags=["消保审查", "产品条款"])
        meta_c = IndexMeta(path="raw/c/index", tags=["消保审查", "产品条款", "销售合规"])
        router = QueryRouter({
            "raw/a/index": meta_a,
            "raw/b/index": meta_b,
            "raw/c/index": meta_c,
        })
        result = router.route_by_tags(["消保审查", "产品条款", "销售合规"], top_k=1)
        assert len(result) == 1
        # meta_c should have highest score (3 exact matches)
        assert result[0] == "raw/c/index"

    def test_route_by_tags_substring_match(self):
        """Substring match between query tag and index tag scores 0.5."""
        meta = IndexMeta(path="raw/idx", tags=["消保审查要点"])
        router = QueryRouter({"raw/idx": meta})
        result = router.route_by_tags(["消保审查"])
        assert "raw/idx" in result

    def test_route_by_tags_name_match(self):
        """Query tag matching index name adds score."""
        meta = IndexMeta(path="raw/idx", tags=[], name="消保审查")
        router = QueryRouter({"raw/idx": meta})
        result = router.route_by_tags(["消保审查"])
        assert "raw/idx" in result

    def test_route_by_tags_source_dir_match(self):
        """Query tag matching source_dir adds score."""
        meta = IndexMeta(path="raw/idx", tags=[], source_dir="insurance")
        router = QueryRouter({"raw/idx": meta})
        result = router.route_by_tags(["insurance"])
        assert "raw/idx" in result

    def test_route_by_tags_multiple_query_tags(self):
        """Multiple query tags match against index tags."""
        meta = IndexMeta(path="raw/insurance/index", tags=["消保审查", "产品条款", "销售合规"])
        router = QueryRouter({"raw/insurance/index": meta})
        result = router.route_by_tags(["消保审查", "产品条款"])
        assert result == ["raw/insurance/index"]

    def test_route_by_tags_mixed_domain_tags(self):
        """Test with realistic insurance domain tags across multiple indexes."""
        meta_review = IndexMeta(
            path="raw/review-points/index",
            tags=["消保审查", "产品条款", "销售合规"],
            name="审查要点",
        )
        meta_product = IndexMeta(
            path="raw/products/index",
            tags=["产品条款", "理赔服务"],
            name="产品库",
        )
        meta_hr = IndexMeta(
            path="raw/hr/index",
            tags=["hr", "人事"],
            name="hr-docs",
        )
        router = QueryRouter({
            "raw/review-points/index": meta_review,
            "raw/products/index": meta_product,
            "raw/hr/index": meta_hr,
        })

        # Searching for "产品条款" should match both review-points and products
        result = router.route_by_tags(["产品条款"])
        assert "raw/review-points/index" in result
        assert "raw/products/index" in result
        # hr should not be in results (no match)
        assert "raw/hr/index" not in result

    def test_route_by_tags_single_index_always_returned(self):
        """Single index is always returned (matched or fallback)."""
        meta = IndexMeta(path="raw/idx", tags=["hr"])
        router = QueryRouter({"raw/idx": meta})
        result = router.route_by_tags(["消保审查"])
        assert result == ["raw/idx"]
