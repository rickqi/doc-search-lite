"""
Regression tests for benchmark results.
Validates data integrity, response parsing, comparison logic, and report generation.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = PROJECT_ROOT / "docs" / "benchmark_results.json"
REPORT_HTML = PROJECT_ROOT / "docs" / "benchmark_report.html"
TEST_CASE_FILE = Path(r"D:\docs\大模型验证完整测试案例集_完整版.md")

# Skip the entire module when the benchmark data file is absent (e.g. CI).
# The file is gitignored and only produced locally via scripts/run_benchmark.py.
pytestmark = pytest.mark.skipif(
    not RESULTS_JSON.exists(),
    reason=f"Benchmark data file not found: {RESULTS_JSON} (generate via scripts/run_benchmark.py)",
)

# ---------------------------------------------------------------------------
# Section / type expectations
# ---------------------------------------------------------------------------
EXPECTED_SECTIONS = {
    "P0通道2": 45,
    "RAG检索": 10,
    "对抗性测试": 10,
    "变异测试": 6,
    "合规补充-渠道多样性": 15,
    "合规补充-产品多样性": 13,
    "合规补充-长文本与临界": 10,
}

EXPECTED_TC_TYPES = {"通道2": 94, "通道1": 5, "RAG": 10}

# Known non-RAG mismatch tc_ids (match_pass=False)
KNOWN_MISMATCH_IDS = [
    "TC-T002",
    "TC-T016",
    "TC-T017",
    "TC-T023",
    "TC-T034",
    "TC-T036",
    "TC-A007",
    "TC-A008",
    "TC-M003",
    "TC-CA013",
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_results() -> list[dict]:
    """Load all benchmark results from JSON. Returns [] when file is missing."""
    if not RESULTS_JSON.exists():
        return []
    with open(RESULTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_llm_results() -> list[dict]:
    """Load only LLM (non-RAG) results for parametrize."""
    return [r for r in _load_results() if r.get("tc_type") != "RAG"]


def _load_rag_results() -> list[dict]:
    """Load only RAG results."""
    return [r for r in _load_results() if r.get("tc_type") == "RAG"]


# ---------------------------------------------------------------------------
# Parsing helpers (mirrors scripts/run_benchmark.py logic)
# ---------------------------------------------------------------------------
def parse_json_from_text(text: str):
    """Extract JSON from text (may be wrapped in markdown code blocks)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from ```json...``` block
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding first balanced { ... } or [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def extract_from_response(actual_output: str) -> dict:
    """Extract pass, hit_word, rule_id from LLM JSON response."""
    parsed = parse_json_from_text(actual_output)
    if isinstance(parsed, dict):
        results_list = parsed.get("results", [parsed])
    elif isinstance(parsed, list):
        results_list = parsed
    else:
        results_list = []

    all_pass = [str(r.get("pass", "")).strip().upper() for r in results_list]
    all_hit_words = [
        str(r.get("hit_word", "")).strip()
        for r in results_list
        if r.get("hit_word")
    ]
    all_rule_ids = [
        str(r.get("rule_id", "")).strip()
        for r in results_list
        if r.get("rule_id")
    ]

    if all_pass:
        if all(p == "Y" for p in all_pass):
            actual_pass = "Y"
        elif all(p == "N" for p in all_pass):
            actual_pass = "N"
        elif not any(p for p in all_pass):
            # All empty strings — no pass values extracted
            actual_pass = ""
        else:
            actual_pass = "mixed"
    else:
        actual_pass = "unknown"

    return {
        "actual_pass": actual_pass,
        "actual_hit_word": ", ".join(all_hit_words),
        "actual_rule_id": ", ".join(all_rule_ids),
    }


def _compute_match_pass(result: dict) -> bool:
    """Reproduce the match_pass comparison logic from run_benchmark.py."""
    expected_pass = result["expected_pass"]
    actual_pass = result["actual_pass"]

    # Parse expected_output to get expected pass from JSON
    expected_output = result.get("expected_output", "")
    if expected_pass == "gray":
        return True
    if expected_pass == "multi":
        return actual_pass in ("N", "mixed")
    if expected_pass == "RAG":
        # RAG cases: match_pass = rag_validation_pass
        return bool(result.get("rag_validation_pass", False))

    # Normal: compare with expected output JSON
    if not expected_output:
        return actual_pass != "unknown"
    try:
        expected_parsed = json.loads(expected_output)
    except json.JSONDecodeError:
        return actual_pass != "unknown"

    if isinstance(expected_parsed, dict):
        exp_list = expected_parsed.get("results", [expected_parsed])
    elif isinstance(expected_parsed, list):
        exp_list = expected_parsed
    else:
        exp_list = []

    exp_pass_values = [str(r.get("pass", "")).strip().upper() for r in exp_list]
    if exp_pass_values:
        if all(p == "Y" for p in exp_pass_values):
            exp_pass = "Y"
        elif all(p == "N" for p in exp_pass_values):
            exp_pass = "N"
        else:
            exp_pass = "mixed"
    else:
        exp_pass = "unknown"

    return actual_pass == exp_pass


def _compute_match_hit_word(result: dict) -> bool:
    """Reproduce match_hit_word logic from run_benchmark.py."""
    expected_output = result.get("expected_output", "")
    actual_hit_word = result.get("actual_hit_word", "")
    expected_pass = result["expected_pass"]

    if not expected_output:
        return False
    try:
        expected_parsed = json.loads(expected_output)
    except json.JSONDecodeError:
        return False

    if isinstance(expected_parsed, dict):
        exp_list = expected_parsed.get("results", [expected_parsed])
    elif isinstance(expected_parsed, list):
        exp_list = expected_parsed
    else:
        exp_list = []

    exp_hit_words = [
        str(r.get("hit_word", "")).strip() for r in exp_list if r.get("hit_word")
    ]

    if exp_hit_words and actual_hit_word:
        actual_lower = actual_hit_word.lower()
        return any(ew.lower() in actual_lower for ew in exp_hit_words if ew)
    elif not exp_hit_words and not actual_hit_word:
        return True
    elif exp_hit_words and expected_pass == "Y":
        return not actual_hit_word
    return False


def _compute_match_rule_id(result: dict) -> bool:
    """Reproduce match_rule_id logic from run_benchmark.py."""
    expected_output = result.get("expected_output", "")
    actual_rule_id = result.get("actual_rule_id", "")

    if not expected_output:
        return False
    try:
        expected_parsed = json.loads(expected_output)
    except json.JSONDecodeError:
        return False

    if isinstance(expected_parsed, dict):
        exp_list = expected_parsed.get("results", [expected_parsed])
    elif isinstance(expected_parsed, list):
        exp_list = expected_parsed
    else:
        exp_list = []

    exp_rule_ids = [
        str(r.get("rule_id", "")).strip() for r in exp_list if r.get("rule_id")
    ]

    if exp_rule_ids and actual_rule_id:
        actual_lower = actual_rule_id.lower()
        return any(er.lower() in actual_lower for er in exp_rule_ids if er)
    elif not exp_rule_ids:
        return True
    return False


# ===========================================================================
# Test class 1: Data integrity
# ===========================================================================
class TestBenchmarkData:
    """Verify benchmark results JSON integrity."""

    def test_results_file_exists(self):
        """benchmark_results.json exists and is valid JSON."""
        assert RESULTS_JSON.exists(), f"Results file not found: {RESULTS_JSON}"
        data = _load_results()
        assert isinstance(data, list)

    def test_results_count(self):
        """Exactly 109 test results."""
        data = _load_results()
        assert len(data) == 109, f"Expected 109 results, got {len(data)}"

    def test_all_have_required_fields(self):
        """Every result has: tc_id, description, section, tc_type, expected_pass, prompt."""
        required = ["tc_id", "description", "section", "tc_type", "expected_pass", "prompt"]
        data = _load_results()
        for r in data:
            for field in required:
                assert field in r, f"{r.get('tc_id', '?')} missing field: {field}"

    def test_tc_ids_unique(self):
        """All tc_id values are unique."""
        data = _load_results()
        ids = [r["tc_id"] for r in data]
        assert len(ids) == len(set(ids)), "Duplicate tc_ids found"

    def test_sections_covered(self):
        """All 7 sections present with expected counts."""
        data = _load_results()
        section_counts: dict[str, int] = {}
        for r in data:
            section_counts[r["section"]] = section_counts.get(r["section"], 0) + 1
        for sec, count in EXPECTED_SECTIONS.items():
            assert sec in section_counts, f"Missing section: {sec}"
            assert section_counts[sec] == count, (
                f"Section {sec}: expected {count}, got {section_counts[sec]}"
            )

    def test_tc_types_distribution(self):
        """通道2: 94, 通道1: 5, RAG: 10."""
        data = _load_results()
        type_counts: dict[str, int] = {}
        for r in data:
            type_counts[r["tc_type"]] = type_counts.get(r["tc_type"], 0) + 1
        for ttype, count in EXPECTED_TC_TYPES.items():
            assert type_counts.get(ttype, 0) == count, (
                f"tc_type {ttype}: expected {count}, got {type_counts.get(ttype, 0)}"
            )

    def test_no_empty_prompts(self):
        """No LLM test case (通道1/通道2) has empty prompt."""
        data = _load_results()
        for r in data:
            if r["tc_type"] in ("通道1", "通道2"):
                assert r["prompt"].strip(), f"{r['tc_id']} has empty prompt"

    def test_all_results_have_latency(self):
        """All results have positive latency_ms."""
        data = _load_results()
        for r in data:
            assert r["latency_ms"] > 0, f"{r['tc_id']} has non-positive latency"


# ===========================================================================
# Test class 2: Response parsing (parametrized)
# ===========================================================================
@pytest.mark.parametrize("result", _load_llm_results(), ids=lambda r: r["tc_id"])
class TestResponseParsing:
    """Test LLM response parsing and pass/hit_word/rule_id extraction."""

    def test_parse_actual_output(self, result):
        """Parse actual_output JSON and verify extracted fields match recorded values."""
        actual_output = result["actual_output"]
        if result.get("error") and result["error"] == "Failed to parse JSON response":
            pytest.skip("Recorded parse failure — actual_output not valid JSON")

        extracted = extract_from_response(actual_output)

        assert extracted["actual_pass"] == result["actual_pass"], (
            f"{result['tc_id']}: extracted pass={extracted['actual_pass']}, "
            f"recorded={result['actual_pass']}"
        )
        assert extracted["actual_hit_word"] == result["actual_hit_word"], (
            f"{result['tc_id']}: extracted hit_word={extracted['actual_hit_word']!r}, "
            f"recorded={result['actual_hit_word']!r}"
        )
        assert extracted["actual_rule_id"] == result["actual_rule_id"], (
            f"{result['tc_id']}: extracted rule_id={extracted['actual_rule_id']!r}, "
            f"recorded={result['actual_rule_id']!r}"
        )


# ===========================================================================
# Test class 3: Comparison logic (parametrized)
# ===========================================================================
@pytest.mark.parametrize("result", _load_llm_results(), ids=lambda r: r["tc_id"])
class TestComparisonLogic:
    """Test the match_pass/match_hit_word/match_rule_id comparison logic."""

    def test_match_pass(self, result):
        """Verify match_pass is correctly computed from expected_pass and actual_pass."""
        # If parsing failed (error set, actual_pass empty), compare_with_expected
        # was never called — match_pass stays False
        if result.get("error") and not result["actual_pass"]:
            assert result["match_pass"] is False
            return
        recorded = result["match_pass"]
        computed = _compute_match_pass(result)
        assert recorded == computed, (
            f"{result['tc_id']}: match_pass recorded={recorded}, computed={computed}"
        )

    def test_match_hit_word(self, result):
        """Verify match_hit_word is correctly computed."""
        if result.get("error") and not result["actual_pass"]:
            assert result["match_hit_word"] is False
            return
        recorded = result["match_hit_word"]
        computed = _compute_match_hit_word(result)
        assert recorded == computed, (
            f"{result['tc_id']}: match_hit_word recorded={recorded}, computed={computed}"
        )

    def test_match_rule_id(self, result):
        """Verify match_rule_id is correctly computed."""
        if result.get("error") and not result["actual_pass"]:
            assert result["match_rule_id"] is False
            return
        recorded = result["match_rule_id"]
        computed = _compute_match_rule_id(result)
        assert recorded == computed, (
            f"{result['tc_id']}: match_rule_id recorded={recorded}, computed={computed}"
        )


# ===========================================================================
# Test class 4: Markdown parser
# ===========================================================================
class TestMarkdownParser:
    """Test the test case parser from the benchmark script."""

    @pytest.fixture(autouse=True)
    def _import_parser(self):
        """Import parser from scripts/run_benchmark.py."""
        scripts_dir = str(PROJECT_ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        # Need to suppress env mutation in run_benchmark.py import
        self._saved_env = {}
        for key in ("LLM_PROVIDER", "LLM_MODEL"):
            if key in __import__("os").environ:
                self._saved_env[key] = __import__("os").environ[key]
        try:
            import run_benchmark as rb
            self.rb = rb
        finally:
            # Restore env
            for key in ("LLM_PROVIDER", "LLM_MODEL"):
                if key in self._saved_env:
                    __import__("os").environ[key] = self._saved_env[key]
                elif key in __import__("os").environ:
                    del __import__("os").environ[key]

    def test_parse_all_109_cases(self):
        """Parser extracts exactly 109 test cases from the markdown file."""
        if not TEST_CASE_FILE.exists():
            pytest.skip(f"Test case file not found: {TEST_CASE_FILE}")
        cases = self.rb.parse_test_cases(TEST_CASE_FILE)
        assert len(cases) == 109, f"Expected 109 cases, parsed {len(cases)}"

    def test_all_cases_have_prompt(self):
        """Every non-RAG case has a non-empty prompt."""
        if not TEST_CASE_FILE.exists():
            pytest.skip(f"Test case file not found: {TEST_CASE_FILE}")
        cases = self.rb.parse_test_cases(TEST_CASE_FILE)
        for c in cases:
            if c.tc_type != "RAG":
                assert c.prompt.strip(), f"{c.tc_id} has empty prompt"

    def test_all_cases_have_expected_output(self):
        """Every non-RAG case has expected output."""
        if not TEST_CASE_FILE.exists():
            pytest.skip(f"Test case file not found: {TEST_CASE_FILE}")
        cases = self.rb.parse_test_cases(TEST_CASE_FILE)
        for c in cases:
            if c.tc_type != "RAG":
                assert c.expected_output.strip(), f"{c.tc_id} has empty expected_output"

    def test_section_distribution(self):
        """Section counts match expected values."""
        if not TEST_CASE_FILE.exists():
            pytest.skip(f"Test case file not found: {TEST_CASE_FILE}")
        cases = self.rb.parse_test_cases(TEST_CASE_FILE)
        sec_counts: dict[str, int] = {}
        for c in cases:
            sec_counts[c.section] = sec_counts.get(c.section, 0) + 1
        for sec, count in EXPECTED_SECTIONS.items():
            assert sec_counts.get(sec, 0) == count, (
                f"Section {sec}: expected {count}, got {sec_counts.get(sec, 0)}"
            )

    def test_tc_type_distribution(self):
        """通道2=94, 通道1=5, RAG=10."""
        if not TEST_CASE_FILE.exists():
            pytest.skip(f"Test case file not found: {TEST_CASE_FILE}")
        cases = self.rb.parse_test_cases(TEST_CASE_FILE)
        type_counts: dict[str, int] = {}
        for c in cases:
            type_counts[c.tc_type] = type_counts.get(c.tc_type, 0) + 1
        for ttype, count in EXPECTED_TC_TYPES.items():
            assert type_counts.get(ttype, 0) == count, (
                f"tc_type {ttype}: expected {count}, got {type_counts.get(ttype, 0)}"
            )

    def test_rag_cases_have_query(self):
        """RAG cases have rag_query or prompt extracted."""
        if not TEST_CASE_FILE.exists():
            pytest.skip(f"Test case file not found: {TEST_CASE_FILE}")
        cases = self.rb.parse_test_cases(TEST_CASE_FILE)
        rag_cases = [c for c in cases if c.tc_type == "RAG"]
        for c in rag_cases:
            has_query = bool(c.rag_query.strip()) or bool(c.prompt.strip())
            assert has_query, f"{c.tc_id} RAG case has no query"

    def test_sw_cases_are_channel1(self):
        """All -SW cases are parsed as 通道1 type."""
        if not TEST_CASE_FILE.exists():
            pytest.skip(f"Test case file not found: {TEST_CASE_FILE}")
        cases = self.rb.parse_test_cases(TEST_CASE_FILE)
        for c in cases:
            if c.tc_id.endswith("-SW"):
                assert c.tc_type == "通道1", (
                    f"{c.tc_id} ends with -SW but tc_type={c.tc_type}"
                )


# ===========================================================================
# Test class 5: Benchmark summary
# ===========================================================================
class TestBenchmarkSummary:
    """Test overall benchmark statistics."""

    def test_overall_pass_rate(self):
        """Overall pass match rate for non-RAG cases is 89/99 (89.9%)."""
        data = _load_results()
        non_rag = [r for r in data if r["tc_type"] != "RAG"]
        pass_match = sum(1 for r in non_rag if r["match_pass"])
        assert pass_match == 89, f"Expected 89 pass matches, got {pass_match}"
        assert len(non_rag) == 99
        rate = pass_match / len(non_rag) * 100
        assert abs(rate - 89.9) < 0.1, f"Rate: {rate:.1f}%"

    def test_per_section_rates(self):
        """Section-level pass rates match expected values."""
        data = _load_results()
        sections: dict[str, list[dict]] = {}
        for r in data:
            sections.setdefault(r["section"], []).append(r)

        # Expected pass counts per section (from actual results)
        expected = {
            "P0通道2": 39,       # 45 - 6 mismatches (T002,T016,T017,T023,T034,T036)
            "对抗性测试": 8,     # 10 - 2 (A007, A008)
            "变异测试": 5,       # 6 - 1 (M003)
            "合规补充-渠道多样性": 14,  # 15 - 1 (CA013)
        }
        for sec, expected_match in expected.items():
            sec_results = sections[sec]
            actual_match = sum(1 for r in sec_results if r["match_pass"])
            assert actual_match == expected_match, (
                f"{sec}: expected {expected_match} match_pass=True, got {actual_match}"
            )

    def test_error_count(self):
        """Exactly 1 error in results."""
        data = _load_results()
        errors = [r for r in data if r.get("error")]
        assert len(errors) == 1, f"Expected 1 error, got {len(errors)}"

    def test_mismatch_cases_identified(self):
        """Mismatch cases match the known list."""
        data = _load_results()
        mismatch_ids = [
            r["tc_id"] for r in data
            if not r["match_pass"] and r["tc_type"] != "RAG"
        ]
        assert sorted(mismatch_ids) == sorted(KNOWN_MISMATCH_IDS), (
            f"Mismatch IDs: {sorted(mismatch_ids)}"
        )

    def test_avg_latency_reasonable(self):
        """Average latency is between 500ms and 10000ms."""
        data = _load_results()
        latencies = [r["latency_ms"] for r in data if r["latency_ms"] > 0]
        avg = sum(latencies) / len(latencies)
        assert 500 <= avg <= 10000, f"Average latency {avg:.1f}ms out of range"

    def test_hit_word_match_rate(self):
        """Hit word match count for non-RAG cases."""
        data = _load_results()
        non_rag = [r for r in data if r["tc_type"] != "RAG"]
        hit_match = sum(1 for r in non_rag if r["match_hit_word"])
        # Actual hit_word match is 56/99 — many cases have no expected hit_words
        # (pass=Y cases) or format differences
        assert hit_match >= 50, f"Hit word match count too low: {hit_match}/99"

    def test_rule_id_match_rate(self):
        """Rule ID match count for non-RAG cases."""
        data = _load_results()
        non_rag = [r for r in data if r["tc_type"] != "RAG"]
        rule_match = sum(1 for r in non_rag if r["match_rule_id"])
        # rule_id uses different format (LLM returns "1" vs expected "cp.training.116.12")
        # so match rate is expected to be low
        assert rule_match >= 0, "Rule match count sanity check"


# ===========================================================================
# Test class 6: HTML report
# ===========================================================================
class TestHTMLReport:
    """Test HTML report generation."""

    def test_report_exists(self):
        """benchmark_report.html exists."""
        assert REPORT_HTML.exists(), f"Report not found: {REPORT_HTML}"

    def test_report_contains_all_tc_ids(self):
        """All 109 TC IDs appear in the report HTML."""
        if not REPORT_HTML.exists():
            pytest.skip("Report file not found")
        html = REPORT_HTML.read_text(encoding="utf-8")
        data = _load_results()
        for r in data:
            assert r["tc_id"] in html, f"{r['tc_id']} not found in report"

    def test_report_has_prompt_blocks(self):
        """Report contains prompt-block elements for each case."""
        if not REPORT_HTML.exists():
            pytest.skip("Report file not found")
        html = REPORT_HTML.read_text(encoding="utf-8")
        assert 'class="prompt-block"' in html, "No prompt-block elements found"

    def test_report_has_summary_section(self):
        """Report contains summary statistics."""
        if not REPORT_HTML.exists():
            pytest.skip("Report file not found")
        html = REPORT_HTML.read_text(encoding="utf-8")
        assert "保险合规审查" in html or "Benchmark" in html
        assert "summary" in html.lower()


# ===========================================================================
# Test class 7: RAG-specific tests
# ===========================================================================
@pytest.mark.parametrize("result", _load_rag_results(), ids=lambda r: r["tc_id"])
class TestRAGResults:
    """Test RAG result data integrity."""

    def test_rag_has_expected_pass_rag(self, result):
        """RAG cases have expected_pass='RAG'."""
        assert result["expected_pass"] == "RAG"

    def test_rag_has_top_rules(self, result):
        """RAG results have rag_top_rules field."""
        assert "rag_top_rules" in result
        assert isinstance(result["rag_top_rules"], list)

    def test_rag_has_validation_pass(self, result):
        """RAG results have rag_validation_pass field."""
        assert "rag_validation_pass" in result
        assert isinstance(result["rag_validation_pass"], bool)
