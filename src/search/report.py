"""Benchmark report generator — Markdown, HTML, JSON.

Generates self-contained reports from :class:`BenchmarkResult` objects.
No external dependencies; all output is pure stdlib.
"""

import json
from dataclasses import asdict

from src.search.benchmark import BenchmarkResult, BenchmarkRunner


class BenchmarkReporter:
    """Generate benchmark reports in multiple formats.

    Usage::

        reporter = BenchmarkReporter()
        md = reporter.generate(result, fmt="markdown")
        html = reporter.generate(result, fmt="html")
        js = reporter.generate(result, fmt="json")
    """

    def generate(self, results: BenchmarkResult, fmt: str = "text") -> str:
        """Generate a report string.

        Args:
            results: BenchmarkResult from BenchmarkRunner.run().
            fmt: One of "text", "markdown" / "md", "html", "json".

        Returns:
            Report as a string.
        """
        if fmt in ("json",):
            return self._json_report(results)
        if fmt in ("html",):
            return self._html_report(results)
        if fmt in ("markdown", "md"):
            return self._markdown_report(results)
        return self._text_report(results)

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _json_report(self, results: BenchmarkResult) -> str:
        """Full JSON dump of all benchmark data."""
        return json.dumps(asdict(results), ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Plain text (console)
    # ------------------------------------------------------------------

    def _text_report(self, results: BenchmarkResult) -> str:
        """Simple text summary for console output."""
        summary = BenchmarkRunner.aggregate_by_mode(results.results)
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("  搜索基准测试报告")
        lines.append("=" * 60)
        lines.append(f"  索引路径: {results.index_path}")
        lines.append(f"  查询数量: {len(results.queries)}")
        lines.append(f"  测试模式: {', '.join(results.modes_tested)}")
        lines.append(f"  总耗时:   {results.total_time:.2f}s")
        lines.append("")

        # Overview table
        lines.append(f"  {'模式':<10s} {'平均延迟':>10s} {'平均命中':>10s} {'平均MRR':>10s} {'成功率':>10s}")
        lines.append(f"  {'-'*50}")
        for mode, stats in summary.items():
            lines.append(
                f"  {mode:<10s} "
                f"{stats['avg_latency']:>9.4f}s "
                f"{stats['avg_hit_rate']:>10.1%} "
                f"{stats['avg_mrr']:>10.4f} "
                f"{stats['avg_result_count']:>8.1f} "
                f"{stats['success_rate']:>9.1%}"
            )
        lines.append("")

        # Per-query details
        for q in results.queries:
            q_results = [r for r in results.results if r.query == q.query]
            if not q_results:
                continue
            lines.append(f"  查询: \"{q.query}\"" + (f"  [{q.category}]" if q.category else ""))
            for mode in results.modes_tested:
                mode_results = [r for r in q_results if r.mode == mode and r.success]
                if not mode_results:
                    lines.append(f"    {mode}: 无成功结果")
                    continue
                avg_lat = sum(r.latency for r in mode_results) / len(mode_results)
                avg_cnt = sum(r.result_count for r in mode_results) / len(mode_results)
                avg_hr = sum(r.hit_rate for r in mode_results) / len(mode_results)
                lines.append(
                    f"    {mode}: 延迟={avg_lat:.4f}s  结果数={avg_cnt:.0f}  命中率={avg_hr:.1%}"
                )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _markdown_report(self, results: BenchmarkResult) -> str:
        """Structured Markdown report."""
        summary = BenchmarkRunner.aggregate_by_mode(results.results)
        lines: list[str] = []

        lines.append("# 搜索基准测试报告")
        lines.append("")
        lines.append(f"- **索引路径**: `{results.index_path}`")
        lines.append(f"- **查询数量**: {len(results.queries)}")
        lines.append(f"- **测试模式**: {', '.join(results.modes_tested)}")
        lines.append(f"- **总耗时**: {results.total_time:.2f}s")
        lines.append("")

        # Overview table
        lines.append("## 总览")
        lines.append("")
        header = "| 模式 | 平均延迟 | 平均命中率 | 平均MRR | 平均结果数 | 成功率 |"
        sep = "|------|----------|-----------|---------|-----------|--------|"
        lines.append(header)
        lines.append(sep)
        for mode, stats in summary.items():
            lines.append(
                f"| {mode} "
                f"| {stats['avg_latency']:.4f}s "
                f"| {stats['avg_hit_rate']:.1%} "
                f"| {stats['avg_mrr']:.4f} "
                f"| {stats['avg_result_count']:.1f} "
                f"| {stats['success_rate']:.1%} |"
            )
        lines.append("")

        # Per-query detailed tables
        lines.append("## 逐查询详情")
        lines.append("")
        for q in results.queries:
            q_results = [r for r in results.results if r.query == q.query]
            if not q_results:
                continue

            cat = f" `[{q.category}]`" if q.category else ""
            lines.append(f"### \"{q.query}\"{cat}")
            lines.append("")
            lines.append("| 模式 | 延迟 | 结果数 | 命中率 | MRR | 状态 |")
            lines.append("|------|------|--------|--------|-----|------|")

            for mode in results.modes_tested:
                mode_results = [r for r in q_results if r.mode == mode]
                for r in mode_results:
                    status = "✅" if r.success else f"❌ {r.error[:30]}"
                    lines.append(
                        f"| {mode} "
                        f"| {r.latency:.4f}s "
                        f"| {r.result_count} "
                        f"| {r.hit_rate:.1%} "
                        f"| {r.mrr:.4f} "
                        f"| {status} |"
                    )
            lines.append("")

        # Conclusions
        lines.append("## 结论")
        lines.append("")
        lines.append(self._conclusions(summary))
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _html_report(self, results: BenchmarkResult) -> str:
        """Self-contained single-file HTML report with inline CSS and SVG."""
        summary = BenchmarkRunner.aggregate_by_mode(results.results)
        modes = results.modes_tested

        # Build overview rows with color coding
        overview_rows = self._html_overview_rows(summary, modes)

        # Build per-query sections
        query_sections = self._html_query_sections(results, modes)

        # Build SVG bar chart for latency
        svg_chart = self._html_latency_chart(summary)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>搜索基准测试报告</title>
<style>
  :root {{
    --bg: #1a1a2e; --fg: #e0e0e0; --table-bg: #16213e;
    --border: #0f3460; --accent: #e94560; --green: #4ecca3;
    --yellow: #f0c040; --red: #e94560; --dim: #888;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--fg); font-family: 'Segoe UI', system-ui, sans-serif; padding: 2rem; line-height: 1.6; }}
  h1 {{ color: var(--accent); margin-bottom: .5rem; }}
  h2 {{ color: var(--yellow); margin: 1.5rem 0 .5rem; border-bottom: 1px solid var(--border); padding-bottom: .25rem; }}
  h3 {{ color: var(--green); margin: 1rem 0 .25rem; }}
  .meta {{ color: var(--dim); margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .5rem 0 1.5rem; }}
  th, td {{ border: 1px solid var(--border); padding: .4rem .7rem; text-align: left; }}
  th {{ background: var(--border); color: var(--fg); }}
  tr:nth-child(even) {{ background: rgba(255,255,255,.03); }}
  .best {{ color: var(--green); font-weight: bold; }}
  .worst {{ color: var(--red); }}
  details {{ margin: .3rem 0 .3rem 1rem; }}
  summary {{ cursor: pointer; color: var(--yellow); }}
  .chart-container {{ margin: 1rem 0 2rem; }}
</style>
</head>
<body>
<h1>🔍 搜索基准测试报告</h1>
<div class="meta">
  <p>索引路径: <code>{results.index_path}</code> &nbsp;|&nbsp;
     查询: {len(results.queries)} 个 &nbsp;|&nbsp;
     模式: {', '.join(modes)} &nbsp;|&nbsp;
     总耗时: {results.total_time:.2f}s</p>
</div>

<h2>总览</h2>
<table>
<thead>
<tr><th>模式</th><th>平均延迟</th><th>平均命中率</th><th>平均MRR</th><th>平均结果数</th><th>成功率</th></tr>
</thead>
<tbody>
{overview_rows}
</tbody>
</table>

<h2>延迟对比</h2>
<div class="chart-container">
{svg_chart}
</div>

<h2>逐查询详情</h2>
{query_sections}

<h2>结论</h2>
<p>{self._conclusions(summary)}</p>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _html_overview_rows(self, summary: dict[str, dict[str, float]], modes: list[str]) -> str:
        """Build <tr> rows for the overview table with green/red coding."""
        if not modes or len(summary) < 2:
            # No comparison possible
            rows = []
            for mode, stats in summary.items():
                rows.append(
                    f"<tr><td>{mode}</td>"
                    f"<td>{stats['avg_latency']:.4f}s</td>"
                    f"<td>{stats['avg_hit_rate']:.1%}</td>"
                    f"<td>{stats['avg_mrr']:.4f}</td>"
                    f"<td>{stats['avg_result_count']:.1f}</td>"
                    f"<td>{stats['success_rate']:.1%}</td></tr>"
                )
            return "\n".join(rows)

        # Determine best/worst for key metrics
        latencies = {m: summary[m]["avg_latency"] for m in modes if m in summary}
        hit_rates = {m: summary[m]["avg_hit_rate"] for m in modes if m in summary}
        mrrs = {m: summary[m]["avg_mrr"] for m in modes if m in summary}

        best_lat = min(latencies, key=lambda m: latencies[m]) if latencies else None
        best_hr = max(hit_rates, key=lambda m: hit_rates[m]) if hit_rates else None
        best_mrr = max(mrrs, key=lambda m: mrrs[m]) if mrrs else None

        rows = []
        for mode, stats in summary.items():
            lat_cls = "best" if mode == best_lat else ("worst" if mode != best_lat and len(summary) > 1 else "")
            hr_cls = "best" if mode == best_hr else ("worst" if mode != best_hr and len(summary) > 1 else "")
            mrr_cls = "best" if mode == best_mrr else ("worst" if mode != best_mrr and len(summary) > 1 else "")

            rows.append(
                f"<tr>"
                f"<td>{mode}</td>"
                f'<td class="{lat_cls}">{stats["avg_latency"]:.4f}s</td>'
                f'<td class="{hr_cls}">{stats["avg_hit_rate"]:.1%}</td>'
                f'<td class="{mrr_cls}">{stats["avg_mrr"]:.4f}</td>'
                f"<td>{stats['avg_result_count']:.1f}</td>"
                f"<td>{stats['success_rate']:.1%}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    def _html_query_sections(self, results: BenchmarkResult, modes: list[str]) -> str:
        """Build collapsible per-query detail sections."""
        sections: list[str] = []
        for q in results.queries:
            q_results = [r for r in results.results if r.query == q.query]
            if not q_results:
                continue

            cat = f" [{q.category}]" if q.category else ""
            rows_html = ""
            for mode in modes:
                mode_results = [r for r in q_results if r.mode == mode]
                for r in mode_results:
                    status = "✅" if r.success else f"❌ {r.error[:40]}"
                    rows_html += (
                        f"<tr><td>{mode}</td>"
                        f"<td>{r.latency:.4f}s</td>"
                        f"<td>{r.result_count}</td>"
                        f"<td>{r.hit_rate:.1%}</td>"
                        f"<td>{r.mrr:.4f}</td>"
                        f"<td>{status}</td></tr>\n"
                    )

            sections.append(
                f"<details><summary><strong>\"{q.query}\"</strong>{cat}</summary>\n"
                f"<table><thead><tr>"
                f"<th>模式</th><th>延迟</th><th>结果数</th>"
                f"<th>命中率</th><th>MRR</th><th>状态</th>"
                f"</tr></thead><tbody>\n{rows_html}</tbody></table>"
                f"</details>\n"
            )
        return "\n".join(sections)

    def _html_latency_chart(self, summary: dict[str, dict[str, float]]) -> str:
        """Inline SVG bar chart comparing latencies across modes."""
        if not summary:
            return "<p>(无数据)</p>"

        modes = list(summary.keys())
        latencies = [summary[m]["avg_latency"] for m in modes]
        max_lat = max(latencies) if latencies else 1.0
        if max_lat == 0:
            max_lat = 1.0

        bar_height = 28
        gap = 10
        label_width = 60
        chart_width = 500
        bar_area = chart_width - label_width - 60

        total_height = len(modes) * (bar_height + gap) + 20

        bars = ""
        colors = ["#4ecca3", "#e94560", "#f0c040", "#6c63ff"]
        for i, (mode, lat) in enumerate(zip(modes, latencies, strict=False)):
            y = i * (bar_height + gap) + 10
            w = max(2, (lat / max_lat) * bar_area)
            color = colors[i % len(colors)]
            bars += (
                f'<text x="0" y="{y + bar_height // 2 + 5}" '
                f'fill="#e0e0e0" font-size="13">{mode}</text>\n'
                f'<rect x="{label_width}" y="{y}" width="{w:.1f}" height="{bar_height}" '
                f'fill="{color}" rx="4"/>\n'
                f'<text x="{label_width + w + 8:.1f}" y="{y + bar_height // 2 + 5}" '
                f'fill="#e0e0e0" font-size="12">{lat:.4f}s</text>\n'
            )

        return (
            f'<svg width="{chart_width}" height="{total_height}" '
            f'xmlns="http://www.w3.org/2000/svg">\n{bars}</svg>'
        )

    # ------------------------------------------------------------------
    # Conclusions
    # ------------------------------------------------------------------

    def _conclusions(self, summary: dict[str, dict[str, float]]) -> str:
        """Generate a short conclusion paragraph from aggregated stats."""
        if not summary:
            return "无测试结果。"

        if len(summary) == 1:
            mode = list(summary.keys())[0]
            s = summary[mode]
            return f"仅测试了 {mode} 模式: 平均延迟 {s['avg_latency']:.4f}s, 命中率 {s['avg_hit_rate']:.1%}。"

        parts: list[str] = []
        # Fastest
        fastest = min(summary, key=lambda m: summary[m]["avg_latency"])
        parts.append(f"{fastest} 延迟最低 ({summary[fastest]['avg_latency']:.4f}s)")

        # Best hit rate
        best_hr = max(summary, key=lambda m: summary[m]["avg_hit_rate"])
        parts.append(f"{best_hr} 命中率最高 ({summary[best_hr]['avg_hit_rate']:.1%})")

        # Best MRR
        best_mrr = max(summary, key=lambda m: summary[m]["avg_mrr"])
        parts.append(f"{best_mrr} MRR 最高 ({summary[best_mrr]['avg_mrr']:.4f})")

        return "对比结果: " + "; ".join(parts) + "。"
