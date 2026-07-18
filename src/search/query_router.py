"""QueryRouter — Route queries to relevant indexes based on metadata and content analysis.

Lightweight keyword-based routing (no LLM by default). Uses:
1. Tag matching (index tags vs query keywords)
2. Domain keyword matching (predefined domain → keyword maps)
3. Name/path-based hints (index name or source_dir in query)

Optional LLM-assisted routing via route_with_llm() when keyword confidence is low.

Usage:
    >>> from src.search.query_router import IndexMeta, QueryRouter
    >>> indexes = {
    ...     "raw/hr/index": IndexMeta(path="raw/hr/index", tags=["hr", "人事"], name="hr-docs"),
    ...     "raw/finance/index": IndexMeta(path="raw/finance/index", tags=["finance", "财务"]),
    ... }
    >>> router = QueryRouter(indexes)
    >>> router.route("年假如何申请")
    ['raw/hr/index']
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexMeta:
    """Metadata for a search index.

    Attributes:
        path: Index path (also used as the key in the router's index dict).
        tags: Domain tags associated with this index (e.g. ["hr", "人事"]).
        profile: Document profile type (legal, technical, faq, general).
        name: Human-readable index name.
        doc_count: Number of documents in this index.
        source_dir: Source directory name (used for path-based matching).
    """

    path: str
    tags: List[str] = field(default_factory=list)
    profile: str = "general"
    name: str = ""
    doc_count: int = 0
    source_dir: str = ""


class QueryRouter:
    """Routes queries to relevant indexes based on keyword matching.

    This is a lightweight router that uses tag matching, domain keyword
    matching, and name/path hints.  An optional :meth:`route_with_llm`
    method can leverage an LLM when keyword confidence is low.

    Args:
        indexes: Dictionary mapping index paths to IndexMeta objects.
    """

    # Domain keyword mapping (expandable)
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "hr": [
            "人事", "员工", "招聘", "薪资", "福利", "请假", "年假",
            "考勤", "入职", "离职", "加班", "hr", "人力资源",
            "培训", "晋升", "调岗", "休假", "绩效", "薪酬",
            # Onboarding (入职)
            "手续", "体检", "转正", "试用期",
            # Compensation (薪酬)
            "工资条", "五险一金", "社保", "公积金", "补缴",
            # Leave (假期)
            "产假", "陪产假", "婚假", "丧假", "调休",
            # Performance (绩效)
            "kpi", "考核", "调薪", "年终奖",
            # Termination (离职)
            "离职证明", "交接", "竞业",
        ],
        "finance": [
            "财务", "报销", "预算", "费用", "发票", "采购", "合同",
            "资金", "税务", "成本", "收入", "支出", "利润",
            # Reimbursement (报销)
            "差旅", "出差", "招待", "交通", "住宿",
            # Budget (预算)
            "预算编制", "费用控制", "审批", "超支",
            # Invoice (发票)
            "增值税", "专票", "普票", "开票", "报税",
            # Payment (付款)
            "付款", "对公", "对私", "转账", "结算",
        ],
        "tech": [
            "技术", "开发", "系统", "api", "代码", "架构", "部署",
            "运维", "服务器", "平台", "工具", "软件", "数据库",
            # Development (开发)
            "前端", "后端", "接口", "联调", "上线",
            # Infrastructure (基础设施)
            "网络", "防火墙", "vpn", "域",
            # Data (数据)
            "备份", "恢复", "迁移", "同步",
        ],
        "legal": [
            "法律", "合规", "政策", "制度", "规定", "条例", "法规",
            "协议", "条款", "知识产权", "保密",
            # Contract (合同)
            "签订", "变更", "终止", "违约", "保密协议",
            # Compliance (合规)
            "监管", "审计", "处罚", "整改", "报送",
        ],
        "admin": [
            "行政", "办公", "资产", "设备", "用车", "会议室",
            "办公用品", "邮件", "通知", "公告",
            # Office (办公)
            "领用", "退还", "申领", "办公电脑", "打印机",
            # Meeting (会议)
            "预定", "视频", "投影",
        ],
        "insurance": [
            # Review (审查)
            "消保", "消保审查", "宣传材料", "条款", "费率", "说明书",
            # Claims (理赔)
            "理赔", "赔付", "理赔申请", "赔款", "免赔额",
            # Product (产品)
            "险种", "保险产品", "承保", "保单", "续保", "退保",
            # Sales (销售)
            "代理人", "销售话术", "营销", "推介",
        ],
    }

    # Colloquial-to-standard term mapping for routing
    _ROUTING_REWRITE_MAP: Dict[str, str] = {
        "多少钱": "费用 价格 报价 成本",
        "怎么办手续": "流程 办理 步骤",
        "能不能": "规定 政策 是否允许",
        "有没有": "制度 标准 规定",
        "需要什么": "要求 条件 材料",
        "去哪里": "地点 窗口 部门",
        "找谁": "负责人 联系人 部门",
        "多久": "期限 时限 时间 周期",
        "合不合法": "合规 法律 法规",
        "扣不扣钱": "扣款 罚款 处罚",
        "怎么算": "计算 公式 标准",
    }

    # Below this keyword score, consider LLM-assisted routing
    LLM_ROUTE_THRESHOLD: float = 1.5

    # LLM routing prompt template
    _LLM_ROUTE_PROMPT: str = (
        "根据用户查询，判断应该在哪个知识库索引中搜索。\n\n"
        "用户查询: {query}\n\n"
        "可用的知识库索引:\n{index_list}\n\n"
        "请选择最相关的索引（可以选多个），输出严格JSON格式（不要markdown包裹）:\n"
        '{{"selected": ["索引名1", "索引名2"], "reason": "选择原因"}}'
    )

    def __init__(self, indexes: Dict[str, IndexMeta]):
        self._indexes: Dict[str, IndexMeta] = indexes
        self._last_max_score: float = 0.0

    @property
    def indexes(self) -> Dict[str, IndexMeta]:
        """Return the registered indexes."""
        return self._indexes

    def rewrite_for_routing(self, query: str) -> str:
        """Rewrite colloquial query for better keyword routing.

        Expands colloquial phrases with their standard equivalents
        to improve routing accuracy without LLM calls.

        Args:
            query: Original user query.

        Returns:
            Query with expanded terms appended. Original query preserved.
        """
        expanded_parts = [query]
        query_lower = query.lower()
        for colloquial, standard in self._ROUTING_REWRITE_MAP.items():
            if colloquial in query_lower:
                expanded_parts.append(standard)
        return " ".join(expanded_parts)

    def route(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Route a query to relevant indexes.

        Args:
            query: Search query string.
            top_k: If set, return at most top_k index paths. None = return all matches.

        Returns:
            List of index paths ordered by relevance (highest first).
            When no index scores positively, returns ALL index paths (fallback).
        """
        if not self._indexes:
            return []

        if not query or not query.strip():
            return list(self._indexes.keys())

        # Rewrite query for better routing
        expanded_query = self.rewrite_for_routing(query)

        # Score each index (use expanded_query instead of raw query)
        scores: Dict[str, float] = {}
        for path, meta in self._indexes.items():
            scores[path] = self._score_index(expanded_query, meta)

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Store max score for route_with_llm() threshold check
        self._last_max_score = max(scores.values()) if scores else 0.0

        # Filter out zero-score indexes only if we have positive scores
        has_positive = any(s > 0 for _, s in ranked)
        if has_positive:
            ranked = [(p, s) for p, s in ranked if s > 0]

        # Apply top_k limit
        if top_k is not None:
            ranked = ranked[:top_k]

        return [path for path, _ in ranked]

    def route_with_llm(
        self,
        query: str,
        top_k: Optional[int] = None,
        llm_client=None,
    ) -> List[str]:
        """Route a query, falling back to LLM when keyword confidence is low.

        First performs keyword routing via :meth:`route`. If the highest keyword
        score is below :attr:`LLM_ROUTE_THRESHOLD` (and an *llm_client* is
        provided), asks the LLM to pick the most relevant indexes.

        Args:
            query: Search query string.
            top_k: If set, return at most *top_k* index paths.
            llm_client: Optional LLM client with a ``chat(messages, temperature,
                max_tokens)`` method returning a response whose ``.content``
                attribute is a string.  When *None*, this method is identical
                to :meth:`route`.

        Returns:
            List of index paths ordered by relevance (highest first).
        """
        # 1. Keyword routing (always runs first)
        keyword_results = self.route(query, top_k)
        max_score = self._last_max_score

        # 2. If keyword confidence is sufficient or no LLM client, return as-is
        if max_score >= self.LLM_ROUTE_THRESHOLD or llm_client is None:
            return keyword_results

        # 3. Build index list string for the LLM prompt
        index_lines: List[str] = []
        for path, meta in self._indexes.items():
            parts = [f"- 名称: {meta.name or Path(path).name}"]
            if meta.tags:
                parts.append(f"  标签: {', '.join(meta.tags)}")
            if meta.source_dir:
                parts.append(f"  来源目录: {meta.source_dir}")
            if meta.doc_count:
                parts.append(f"  文档数: {meta.doc_count}")
            index_lines.append("\n".join(parts))

        index_list = "\n\n".join(index_lines)
        prompt = self._LLM_ROUTE_PROMPT.format(query=query, index_list=index_list)

        # 4. Call LLM (lazy import to avoid circular deps)
        try:
            from src.agent.llm_client import ChatMessage  # noqa: WPS433

            messages = [ChatMessage(role="user", content=prompt)]
            response = llm_client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=200,
            )
            raw = response.content.strip()
        except Exception:
            logger.warning("LLM routing call failed, falling back to keyword results")
            return keyword_results

        # 5. Parse JSON response (handle markdown code blocks)
        try:
            # Strip markdown code fences if present
            cleaned = raw
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first and last lines (``` fences)
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)
            selected_names: List[str] = data.get("selected", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("LLM routing response parse failed: %s", raw[:100])
            return keyword_results

        if not selected_names:
            return keyword_results

        # 6. Match LLM-selected names to index paths
        path_map: Dict[str, str] = {}
        for path, meta in self._indexes.items():
            dir_name = Path(path).name
            if meta.name:
                path_map[meta.name.lower()] = path
            path_map[dir_name.lower()] = path

        matched: List[str] = []
        for name in selected_names:
            key = name.lower().strip()
            if key in path_map and path_map[key] not in matched:
                matched.append(path_map[key])

        if not matched:
            logger.warning("LLM routing: no index matches for %s", selected_names)
            return keyword_results

        logger.info(
            "LLM routing: query='%s...', keyword_score=%.1f, selected=%s",
            query[:30],
            max_score,
            matched,
        )

        # Apply top_k limit
        if top_k is not None:
            matched = matched[:top_k]

        return matched

    def _score_index(self, query: str, meta: IndexMeta) -> float:
        """Score an index's relevance to a query.

        Scoring weights:
        - Tag exact match: 2.0 per match
        - Tag partial (single char overlap): 0.5
        - Domain keyword + tag match: 1.5 per domain
        - Name match: 1.0
        - Source dir match: 0.5
        """
        score = 0.0
        query_lower = query.lower()

        # 1. Tag matching (highest weight)
        for tag in meta.tags:
            tag_lower = tag.lower()
            if tag_lower in query_lower:
                score += 2.0
            elif len(tag_lower) > 1:
                # Partial match: any multi-char tag substring in query
                for i in range(len(tag_lower)):
                    for j in range(i + 2, len(tag_lower) + 1):
                        if tag_lower[i:j] in query_lower:
                            score += 0.3
                            break
                    else:
                        continue
                    break

        # 2. Domain keyword matching
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            domain_match = any(kw in query_lower for kw in keywords)
            tag_lower_list = [t.lower() for t in meta.tags]
            tag_match = domain in tag_lower_list
            if domain_match and tag_match:
                score += 1.5

        # 3. Name matching
        if meta.name and meta.name.lower() in query_lower:
            score += 1.0

        # 4. Source dir matching
        if meta.source_dir and meta.source_dir.lower() in query_lower:
            score += 0.5

        return score

    def route_by_tags(self, query_tags: List[str], top_k: Optional[int] = None) -> List[str]:
        """Route based on explicit tags (for recall scenario).

        Instead of matching query text → index tags (like route()), this matches
        explicit document tags → index metadata tags. Designed for the recall
        scenario where input is a set of document labels/tags and output should
        be the most relevant indexes.

        Scoring:
        - Exact tag match (case-insensitive): 2.0 per match
        - Partial tag overlap (shared substring >= 2 chars): 0.5 per pair

        Args:
            query_tags: List of tags to match (e.g. ["消保审查", "产品条款"]).
            top_k: If set, return at most top_k index paths. None = return all matches.

        Returns:
            List of index paths ordered by tag relevance (highest first).
            When no index scores positively, returns ALL index paths (fallback).
        """
        if not self._indexes:
            return []

        if not query_tags:
            return list(self._indexes.keys())

        # Normalize query tags to lowercase
        query_tags_lower = [t.lower() for t in query_tags]

        # Score each index by tag overlap
        scores: Dict[str, float] = {}
        for path, meta in self._indexes.items():
            score = 0.0
            index_tags_lower = [t.lower() for t in meta.tags]

            # 1. Exact match scoring
            for qt in query_tags_lower:
                for it in index_tags_lower:
                    if qt == it:
                        score += 2.0
                    elif qt in it or it in qt:
                        # Substring match: one contains the other
                        score += 0.5

            # 2. Cross-match: query tag keywords against index metadata
            # (name, source_dir) for bonus scoring
            for qt in query_tags_lower:
                if meta.name and qt in meta.name.lower():
                    score += 1.0
                if meta.source_dir and qt in meta.source_dir.lower():
                    score += 0.5

            scores[path] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out zero-score indexes only if we have positive scores
        has_positive = any(s > 0 for _, s in ranked)
        if has_positive:
            ranked = [(p, s) for p, s in ranked if s > 0]

        # Apply top_k limit
        if top_k is not None:
            ranked = ranked[:top_k]

        return [path for path, _ in ranked]
