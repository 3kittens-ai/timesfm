#!/usr/bin/env python3
"""对比最近两份九研预测 JSON，并生成 Markdown 差异报告。"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FILE_PATTERN = "jiuyan_forecasts_*.json"
DATE_RE = re.compile(r"jiuyan_forecasts_(\d{8})\.json$")


@dataclass(frozen=True)
class ForecastFile:
    date_str: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比最近两份预测 JSON，输出 Markdown 差异报告")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"预测 JSON 所在目录，默认 {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="报告输出目录，默认与 output-dir 相同",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="报告中展示的增减幅 Top N，默认 20",
    )
    return parser.parse_args()


def discover_forecast_files(output_dir: Path) -> list[ForecastFile]:
    files: list[ForecastFile] = []
    for path in output_dir.glob(FILE_PATTERN):
        match = DATE_RE.fullmatch(path.name)
        if not match:
            continue
        files.append(ForecastFile(date_str=match.group(1), path=path))
    return sorted(files, key=lambda item: item.date_str, reverse=True)


def filter_files_in_latest_month(files: list[ForecastFile]) -> list[ForecastFile]:
    if not files:
        return []
    latest_month = files[0].date_str[:6]
    return [item for item in files if item.date_str.startswith(latest_month)]


def load_forecast_json(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"预测文件格式不正确: {path}")
    return data


def safe_pct_change(old_value: float, new_value: float) -> str:
    if math.isclose(old_value, 0.0, abs_tol=1e-9):
        return "N/A" if math.isclose(new_value, 0.0, abs_tol=1e-9) else "新增"
    return f"{(new_value - old_value) / old_value * 100:+.2f}%"


def summarize_sku_diff_for_month(old_payload: dict, new_payload: dict, target_month: str) -> dict | None:
    old_map = {
        str(month): float(value or 0)
        for month, value in zip(old_payload.get("months") or [], old_payload.get("forecast") or [])
    }
    new_map = {
        str(month): float(value or 0)
        for month, value in zip(new_payload.get("months") or [], new_payload.get("forecast") or [])
    }
    if target_month not in old_map or target_month not in new_map:
        return None

    old_value = old_map[target_month]
    new_value = new_map[target_month]
    delta = new_value - old_value
    return {
        "target_month": target_month,
        "old_total": old_value,
        "new_total": new_value,
        "delta": delta,
        "abs_delta": abs(delta),
        "pct_change": safe_pct_change(old_value, new_value),
    }


def format_number(value: float) -> str:
    return f"{value:,.1f}"


def render_rank_table(rows: list[dict], title: str, old_date: str, new_date: str, target_month: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"| SKU | {old_date} {target_month} 月预测量 | {new_date} {target_month} 月预测量 | 差值 | 变化幅度 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| - | - | - | - | - |")
        lines.append("")
        return lines

    for row in rows:
        lines.append(
            f"| {row['sku']} | {format_number(row['old_total'])} | "
            f"{format_number(row['new_total'])} | {format_number(row['delta'])} | {row['pct_change']} |"
        )
    lines.append("")
    return lines


def generate_report(old_file: ForecastFile, new_file: ForecastFile, top_n: int) -> tuple[str, Path]:
    old_data = load_forecast_json(old_file.path)
    new_data = load_forecast_json(new_file.path)
    target_month = f"{new_file.date_str[:4]}-{new_file.date_str[4:6]}"

    old_skus = set(old_data)
    new_skus = set(new_data)
    common_skus = old_skus & new_skus
    old_only = sorted(old_skus - new_skus)
    new_only = sorted(new_skus - old_skus)

    target_month_diffs = []
    changed_count_target_month = 0
    old_total_target_month = 0.0
    new_total_target_month = 0.0

    for sku in sorted(common_skus):
        target_month_summary = summarize_sku_diff_for_month(old_data[sku], new_data[sku], target_month)
        if target_month_summary is not None:
            target_month_summary["sku"] = sku
            target_month_diffs.append(target_month_summary)
            old_total_target_month += target_month_summary["old_total"]
            new_total_target_month += target_month_summary["new_total"]
            if not math.isclose(target_month_summary["delta"], 0.0, abs_tol=1e-9):
                changed_count_target_month += 1

    target_month_diffs.sort(key=lambda item: item["delta"], reverse=True)
    top_increase = target_month_diffs[:top_n]
    top_decrease = sorted(target_month_diffs, key=lambda item: item["delta"])[:top_n]
    abs_deltas = [item["abs_delta"] for item in target_month_diffs]
    avg_abs_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
    median_abs_delta = median(abs_deltas) if abs_deltas else 0.0

    report_lines = [
        f"# {new_file.date_str} vs {old_file.date_str} 预测差异报告",
        "",
        "## 文件概况",
        "",
        f"- 新文件销售数据截止日期: `{new_file.date_str}`",
        f"- 旧文件销售数据截止日期: `{old_file.date_str}`",
        "",
        "## SKU 覆盖情况",
        "",
        f"- `{new_file.date_str}` 总 SKU 数: **{len(new_skus)}**",
        f"- `{old_file.date_str}` 总 SKU 数: **{len(old_skus)}**",
        f"- 重合 SKU 数: **{len(common_skus)}**",
        f"- 仅 `{new_file.date_str}` 存在的 SKU 数: **{len(new_only)}**",
        f"- 仅 `{old_file.date_str}` 存在的 SKU 数: **{len(old_only)}**",
        "",
        "## 重合 SKU 总体差异",
        "",
        f"- 纳入差异分析的重合 SKU 数: **{len(target_month_diffs)}**",
        f"- 发生预测变化的 SKU 数: **{changed_count_target_month}**",
        f"- 重合 SKU 在 `{old_file.date_str}` 中 `{target_month}` 月预测总量: **{format_number(old_total_target_month)}**",
        f"- 重合 SKU 在 `{new_file.date_str}` 中 `{target_month}` 月预测总量: **{format_number(new_total_target_month)}**",
        f"- 总差值: **{format_number(new_total_target_month - old_total_target_month)}**",
        f"- 总体变化幅度: **{safe_pct_change(old_total_target_month, new_total_target_month)}**",
        f"- 单 SKU 平均绝对变化量: **{format_number(avg_abs_delta)}**",
        f"- 单 SKU 绝对变化量中位数: **{format_number(median_abs_delta)}**",
    ]

    report_lines.append("")
    report_lines.extend(
        render_rank_table(top_increase, f"重合 SKU 增幅 Top {top_n}", old_file.date_str, new_file.date_str, target_month)
    )
    report_lines.extend(
        render_rank_table(top_decrease, f"重合 SKU 降幅 Top {top_n}", old_file.date_str, new_file.date_str, target_month)
    )

    report_name = f"{new_file.date_str}_{old_file.date_str}_diff_report.md"
    return "\n".join(report_lines), Path(report_name)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    report_dir = (args.report_dir or output_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    forecast_files = filter_files_in_latest_month(discover_forecast_files(output_dir))
    if len(forecast_files) < 2:
        raise SystemExit("本月还未生成至少 2 份AI生产计划，无法对比AI生产计划的差异")

    new_file, old_file = forecast_files[0], forecast_files[1]
    report_text, report_name = generate_report(old_file=old_file, new_file=new_file, top_n=max(1, args.top_n))
    report_path = report_dir / report_name
    report_path.write_text(report_text, encoding="utf-8")

    print(f"已对比: {new_file.path.name} vs {old_file.path.name}")
    print(f"报告已生成: {report_path}")


if __name__ == "__main__":
    main()
