"""Usage report generator for doc-search.

Exports statistics in JSON, CSV, Markdown, and HTML formats.
Uses only stdlib: json, csv, html.
"""

import csv
import html as html_mod
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API 用量统计报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 2rem auto; max-width: 960px; color: #1a1a1a; background: #fff;
       line-height: 1.6; }}
h1 {{ border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; }}
h2 {{ color: #333; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #f5f5f5; font-weight: 600; text-align: center; }}
td:first-child {{ text-align: left; }}
tr:nth-child(even) {{ background: #fafafa; }}
.generated {{ color: #888; font-size: 0.85rem; margin-top: 2rem; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


class StatsReporter:
    """Generate usage reports in multiple formats.

    All export methods return the content as a string.
    If output_path is provided, the content is also written to the file.
    """

    def __init__(self, db: Any):
        """Initialize StatsReporter.

        Args:
            db: ConvertDB instance (must be open)
        """
        self._db = db

    def generate_summary(
        self, source_dir: str = None, days: int = None
    ) -> dict:
        """Collect all statistics into a summary dict.

        Args:
            source_dir: Optional source directory filter
            days: Optional limit to last N days

        Returns:
            Dict with summary, daily, and models sections
        """
        summary_data = self._db.get_token_usage_summary(
            source_dir=source_dir, days=days
        )

        daily_rows = self._db.get_token_usage_daily(
            days=days or 30, source_dir=source_dir
        )

        model_rows = self._db.get_token_usage_by_model(
            source_dir=source_dir, days=days
        )

        return {
            "summary": {
                "by_type": summary_data.get("by_type", {}),
                "total": summary_data.get("total", {}),
            },
            "daily": daily_rows,
            "models": model_rows,
        }

    def export_json(self, data: dict, output_path: Path = None) -> str:
        """Export statistics as JSON.

        Args:
            data: Summary dict from generate_summary()
            output_path: Optional file path to write

        Returns:
            JSON string
        """
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        return content

    def export_csv(self, data: dict, output_path: Path = None) -> str:
        """Export statistics as CSV.

        Args:
            data: Summary dict from generate_summary()
            output_path: Optional file path to write

        Returns:
            CSV string
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "section", "key", "call_count", "input_tokens",
            "output_tokens", "total_tokens", "cost_millicents",
        ])

        # Summary by type
        by_type = data.get("summary", {}).get("by_type", {})
        for ct, row in sorted(by_type.items()):
            writer.writerow([
                "summary", ct,
                row.get("call_count", 0),
                row.get("input_tokens", 0),
                row.get("output_tokens", 0),
                row.get("total_tokens", 0),
                row.get("cost_millicents", 0),
            ])

        # Daily
        for row in data.get("daily", []):
            writer.writerow([
                "daily", row.get("date", ""),
                row.get("call_count", 0),
                row.get("input_tokens", 0),
                row.get("output_tokens", 0),
                row.get("total_tokens", 0),
                row.get("cost_millicents", 0),
            ])

        # Models
        for row in data.get("models", []):
            writer.writerow([
                "model", row.get("model", ""),
                row.get("call_count", 0),
                row.get("input_tokens", 0),
                row.get("output_tokens", 0),
                row.get("total_tokens", 0),
                row.get("cost_millicents", 0),
            ])

        content = output.getvalue()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        return content

    def export_markdown(self, data: dict, output_path: Path = None) -> str:
        """Export statistics as Markdown.

        Args:
            data: Summary dict from generate_summary()
            output_path: Optional file path to write

        Returns:
            Markdown string
        """
        lines = ["# API 用量统计报告\n"]

        # Summary section
        lines.append("## 总体汇总\n")
        lines.append(
            "| 类型 | 调用次数 | Input | Output | Total | 费用(¥) |"
        )
        lines.append(
            "|------|---------|--------|---------|--------|---------|"
        )
        by_type = data.get("summary", {}).get("by_type", {})
        for ct in sorted(by_type.keys()):
            r = by_type[ct]
            lines.append(
                f"| {ct} | {r.get('call_count', 0)} "
                f"| {r.get('input_tokens', 0):,} "
                f"| {r.get('output_tokens', 0):,} "
                f"| {r.get('total_tokens', 0):,} "
                f"| ¥{r.get('cost_millicents', 0) / 100000:.4f} |"
            )

        total = data.get("summary", {}).get("total", {})
        if total:
            lines.append(
                f"| **合计** | {total.get('call_count', 0)} "
                f"| {total.get('input_tokens', 0):,} "
                f"| {total.get('output_tokens', 0):,} "
                f"| {total.get('total_tokens', 0):,} "
                f"| ¥{total.get('cost_millicents', 0) / 100000:.4f} |"
            )

        # Daily section
        daily = data.get("daily", [])
        if daily:
            lines.append("\n## 每日趋势\n")
            lines.append(
                "| 日期 | 调用 | Input | Output | Total | 费用(¥) |"
            )
            lines.append(
                "|------|------|--------|---------|--------|---------|"
            )
            for r in daily:
                lines.append(
                    f"| {r.get('date', '')} "
                    f"| {r.get('call_count', 0)} "
                    f"| {r.get('input_tokens', 0):,} "
                    f"| {r.get('output_tokens', 0):,} "
                    f"| {r.get('total_tokens', 0):,} "
                    f"| ¥{r.get('cost_millicents', 0) / 100000:.4f} |"
                )

        # Models section
        models = data.get("models", [])
        if models:
            lines.append("\n## 模型统计\n")
            lines.append(
                "| 模型 | 调用 | Input | Output | Total | 费用(¥) |"
            )
            lines.append(
                "|------|------|--------|---------|--------|---------|"
            )
            for r in models:
                lines.append(
                    f"| {r.get('model', '')} "
                    f"| {r.get('call_count', 0)} "
                    f"| {r.get('input_tokens', 0):,} "
                    f"| {r.get('output_tokens', 0):,} "
                    f"| {r.get('total_tokens', 0):,} "
                    f"| ¥{r.get('cost_millicents', 0) / 100000:.4f} |"
                )

        content = "\n".join(lines)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        return content

    def export_html(self, data: dict, output_path: Path = None) -> str:
        """Export statistics as a standalone HTML file with embedded CSS.

        Args:
            data: Summary dict from generate_summary()
            output_path: Optional file path to write

        Returns:
            HTML string
        """
        parts = []
        parts.append("<h1>API 用量统计报告</h1>")
        parts.append(
            f'<p class="generated">生成时间: '
            f'{html_mod.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>'
        )

        # Summary table
        parts.append("<h2>总体汇总</h2>")
        parts.append("<table>")
        parts.append(
            "<tr><th>类型</th><th>调用次数</th><th>Input</th>"
            "<th>Output</th><th>Total</th><th>费用(¥)</th></tr>"
        )
        by_type = data.get("summary", {}).get("by_type", {})
        for ct in sorted(by_type.keys()):
            r = by_type[ct]
            parts.append(
                f"<tr><td>{html_mod.escape(ct)}</td>"
                f"<td>{r.get('call_count', 0)}</td>"
                f"<td>{r.get('input_tokens', 0):,}</td>"
                f"<td>{r.get('output_tokens', 0):,}</td>"
                f"<td>{r.get('total_tokens', 0):,}</td>"
                f"<td>¥{r.get('cost_millicents', 0) / 100000:.4f}</td></tr>"
            )
        total = data.get("summary", {}).get("total", {})
        if total:
            parts.append(
                f"<tr><td><strong>合计</strong></td>"
                f"<td>{total.get('call_count', 0)}</td>"
                f"<td>{total.get('input_tokens', 0):,}</td>"
                f"<td>{total.get('output_tokens', 0):,}</td>"
                f"<td>{total.get('total_tokens', 0):,}</td>"
                f"<td>¥{total.get('cost_millicents', 0) / 100000:.4f}</td></tr>"
            )
        parts.append("</table>")

        # Daily table
        daily = data.get("daily", [])
        if daily:
            parts.append("<h2>每日趋势</h2>")
            parts.append("<table>")
            parts.append(
                "<tr><th>日期</th><th>调用</th><th>Input</th>"
                "<th>Output</th><th>Total</th><th>费用(¥)</th></tr>"
            )
            for r in daily:
                parts.append(
                    f"<tr><td>{html_mod.escape(r.get('date', ''))}</td>"
                    f"<td>{r.get('call_count', 0)}</td>"
                    f"<td>{r.get('input_tokens', 0):,}</td>"
                    f"<td>{r.get('output_tokens', 0):,}</td>"
                    f"<td>{r.get('total_tokens', 0):,}</td>"
                    f"<td>¥{r.get('cost_millicents', 0) / 100000:.4f}</td></tr>"
                )
            parts.append("</table>")

        # Models table
        models = data.get("models", [])
        if models:
            parts.append("<h2>模型统计</h2>")
            parts.append("<table>")
            parts.append(
                "<tr><th>模型</th><th>调用</th><th>Input</th>"
                "<th>Output</th><th>Total</th><th>费用(¥)</th></tr>"
            )
            for r in models:
                parts.append(
                    f"<tr><td>{html_mod.escape(r.get('model', ''))}</td>"
                    f"<td>{r.get('call_count', 0)}</td>"
                    f"<td>{r.get('input_tokens', 0):,}</td>"
                    f"<td>{r.get('output_tokens', 0):,}</td>"
                    f"<td>{r.get('total_tokens', 0):,}</td>"
                    f"<td>¥{r.get('cost_millicents', 0) / 100000:.4f}</td></tr>"
                )
            parts.append("</table>")

        body = "\n".join(parts)
        content = _HTML_TEMPLATE.format(body=body)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        return content
