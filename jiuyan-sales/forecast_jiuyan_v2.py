#!/usr/bin/env python3
"""
forecast_jiuyan.py — 九研 SKU 月度销量 × TimesFM 2.5 零样本预测

从 sales_filtered.sqlite 的 v_training_base 视图提取月度 SKU 销量，
使用 Google TimesFM 基础模型进行零样本预测，输出未来 N 个月的
点预测 + 80% 预测区间。

用法:
    # 基础预测 (全部 SKU, 12 个月)
    python forecast_jiuyan.py

    # 指定预测步长和最低历史月数
    python forecast_jiuyan.py --horizon 6 --min-months 24

    # 只预测指定 SKU
    python forecast_jiuyan.py --skus 6974220210331,6974220210324

    # 输出为 CSV (默认 JSON)
    python forecast_jiuyan.py --format csv --output results.csv

    # 跳过系统检查 (不推荐)
    python forecast_jiuyan.py --skip-check

    # 限制最大 SKU 数量 (快速试跑)
    python forecast_jiuyan.py --max-skus 50
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import calendar
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 优先使用当前仓库源码，避免落到虚拟环境中的旧版 timesfm 包。
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
DB_PATH = "/Users/andychan/Documents/jiuyan/sales_filtered_database/sales_filtered.sqlite"
DEFAULT_HORIZON = 12
DEFAULT_MIN_MONTHS = 12
CHUNK_SIZE = 200  # 每批次处理的 SKU 数量，防止内存溢出


def build_default_output_path(
    output_dir: Path,
    latest_sale_ts: pd.Timestamp | None,
) -> Path:
    """按实际销售截止日期生成默认 JSON 文件名。"""
    if latest_sale_ts is None:
        return output_dir / "jiuyan_forecasts.json"
    return output_dir / f"jiuyan_forecasts_{latest_sale_ts.strftime('%Y%m%d')}.json"

def fetch_daily_sales(db_path: str, sku: str, end_date: str, days: int = 400) -> np.ndarray:
    """从 sales 表抓取指定 SKU 的日销量背景。"""
    conn = sqlite3.connect(db_path)
    end_ts = pd.Timestamp(end_date)
    start_ts = end_ts - timedelta(days=days-1)
    
    query = f"""
        SELECT sale_date, SUM(sales_volume) as daily_vol
        FROM sales
        WHERE barcode = '{sku}' AND sale_date BETWEEN '{start_ts.strftime('%Y-%m-%d')}' AND '{end_ts.strftime('%Y-%m-%d')}'
        GROUP BY sale_date
    """
    df_daily = pd.read_sql(query, conn)
    conn.close()
    
    if df_daily.empty:
        return np.zeros(days, dtype=np.float32)
        
    df_daily["sale_date"] = pd.to_datetime(df_daily["sale_date"])
    full_range = pd.date_range(start=start_ts, end=end_ts, freq="D")
    full_df = pd.DataFrame({"sale_date": full_range})
    merged = full_df.merge(df_daily, on="sale_date", how="left")
    return merged["daily_vol"].fillna(0).values.astype(np.float32)

# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────
def load_data(
    db_path: str = DB_PATH,
    min_months: int = DEFAULT_MIN_MONTHS,
    sku_filter: list[str] | None = None,
    max_skus: int | None = None,
    refine: bool = False,
    daily_days: int = 400,
) -> tuple[list[np.ndarray], list[str], pd.Timestamp, pd.Timestamp | None, list[np.ndarray]]:
    """从 SQLite 加载月度 SKU 销量，填充缺失月份，返回 TimesFM 输入格式。

    Returns:
        inputs: 每个元素是一个 SKU 的月度销量一维数组
        sku_index: 与 inputs 一一对应的 SKU 编码列表
        last_date: 数据中最后一个完结月的日期
        latest_sale_ts: 数据库中最新的销售日期 (用于 refine 逻辑)
        daily_contexts: 日度数据数组的列表
    """
    if not Path(db_path).exists():
        print(f"🛑 数据库不存在: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    # 构建查询
    query = """
        SELECT variant_key, year_month, monthly_qty
        FROM v_training_base
    """
    if sku_filter:
        placeholders = ",".join("?" for _ in sku_filter)
        query += f" WHERE variant_key IN ({placeholders})"
        df = pd.read_sql(query, conn, params=sku_filter)
    else:
        df = pd.read_sql(query, conn)

    if df.empty:
        conn.close()
        print("🛑 查询结果为空，请检查数据库或 SKU 编码。")
        sys.exit(1)

    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    global_max = df["date"].max()

    latest_sale_date = pd.read_sql(
        "SELECT MAX(sale_date) AS latest_sale_date FROM sales",
        conn,
    ).iloc[0, 0]
    conn.close()

    latest_sale_ts = pd.to_datetime(latest_sale_date) if latest_sale_date else None
    if latest_sale_ts is not None and not latest_sale_ts.is_month_end:
        incomplete_month = latest_sale_ts.to_period("M").to_timestamp()
        if incomplete_month <= global_max:
            if refine:
                print(
                    "   ✨ 检测到未完结月份，且开启了日度修正(--refine): "
                    f"{incomplete_month.strftime('%Y-%m')} "
                    f"(最新销售日期 {latest_sale_ts.strftime('%Y-%m-%d')})"
                )
            else:
                print(
                    "   ⚠️ 检测到未完结月份，已自动忽略 (未使用--refine): "
                    f"{incomplete_month.strftime('%Y-%m')} "
                    f"(最新销售日期 {latest_sale_ts.strftime('%Y-%m-%d')})"
                )
            # 在两种情况下我们都忽略它作为 *月度基线* 数据
            df = df[df["date"] < incomplete_month].copy()

            if df.empty:
                print("🛑 过滤未完结月份后没有可用月度历史数据。")
                sys.exit(1)

            global_max = df["date"].max()

    print(f"   数据范围: {df['date'].min().strftime('%Y-%m')} → {global_max.strftime('%Y-%m')}")
    if latest_sale_ts is not None:
        print(f"   最新销售流水日期: {latest_sale_ts.strftime('%Y-%m-%d')}")
    print(f"   原始 SKU 数: {df['variant_key'].nunique()}")

    # ── 按 SKU 分组，填充缺失月份为 0 ──
    inputs = []
    sku_index = []

    grouped = df.groupby("variant_key")
    for sku, group in grouped:
        group = group.sort_values("date")
        start = group["date"].min()

        # 构建从首次出现到全局结束的完整月份序列
        full_range = pd.date_range(start=start, end=global_max, freq="MS")
        full_df = pd.DataFrame({"date": full_range})
        merged = full_df.merge(
            group[["date", "monthly_qty"]], on="date", how="left"
        )
        merged["monthly_qty"] = merged["monthly_qty"].fillna(0)

        series = merged["monthly_qty"].values.astype(np.float32)

        if len(series) >= min_months:
            inputs.append(series)
            sku_index.append(sku)

    if max_skus and len(inputs) > max_skus:
        inputs = inputs[:max_skus]
        sku_index = sku_index[:max_skus]
        print(f"   ⚠️ 已限制为前 {max_skus} 个 SKU")

    if not inputs:
        print(f"🛑 没有 SKU 满足 ≥ {min_months} 个月的条件。")
        sys.exit(1)

    # 如果开启修正，抓取日度数据
    daily_contexts = []
    if refine and latest_sale_ts is not None and not latest_sale_ts.is_month_end:
         print(f"   正在抓取 {len(sku_index)} 个 SKU 的日销量背景 ({daily_days} 天)...")
         cutoff_date_str = latest_sale_ts.strftime('%Y-%m-%d')
         for sku in sku_index:
             daily_contexts.append(fetch_daily_sales(db_path, str(sku), cutoff_date_str, daily_days))

    lengths = [len(s) for s in inputs]
    print(f"   有效 SKU 数: {len(inputs)}")
    print(f"   序列长度: {min(lengths)} ~ {max(lengths)} 月")

    return inputs, sku_index, global_max, latest_sale_ts, daily_contexts


# ──────────────────────────────────────────────
# 系统检查
# ──────────────────────────────────────────────
def run_preflight(num_series: int, context_length: int, horizon: int) -> None:
    """运行系统预检。"""
    script_dir = REPO_ROOT / "timesfm-forecasting" / "scripts"
    sys.path.insert(0, str(script_dir))

    try:
        from check_system import run_checks  # type: ignore

        report = run_checks("v2.5")
        if not report.passed:
            print(f"\n🛑 系统预检未通过: {report.verdict_detail}")
            print("   请运行 python timesfm-forecasting/scripts/check_system.py 查看详情")
            sys.exit(1)
        print(f"   {report.verdict}")
        print(f"   {report.verdict_detail}")
    except ImportError:
        print("   ⚠️ 跳过系统预检 (check_system 模块不可用)")
    finally:
        if str(script_dir) in sys.path:
            sys.path.remove(str(script_dir))


# ──────────────────────────────────────────────
# 模型加载 & 预测
# ──────────────────────────────────────────────
def load_model(horizon: int, max_context: int = 1024, batch_size: int = 32):
    """加载 TimesFM 2.5 PyTorch 模型并编译。"""
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")

    print(f"   正在加载 TimesFM 2.5 (200M) ...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    print(f"   正在编译模型 (batch_size={batch_size}, max_context={max_context})...")
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max(256, horizon),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=batch_size,
        )
    )

    return model


def run_forecast(
    model, inputs: list[np.ndarray], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """分批次运行预测，防止内存溢出。"""
    all_point = []
    all_quant = []
    total = len(inputs)

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        batch = inputs[start:end]
        print(f"   预测中: [{start + 1}~{end}] / {total} SKU ...", end="", flush=True)

        t0 = time.time()
        # TimesFM 2.5 doesn't need freq
        point, quantiles = model.forecast(horizon=horizon, inputs=batch)
        elapsed = time.time() - t0

        all_point.append(point)
        all_quant.append(quantiles)
        print(f" {elapsed:.1f}s")

    return np.concatenate(all_point, axis=0), np.concatenate(all_quant, axis=0)


# ──────────────────────────────────────────────
# 结果输出
# ──────────────────────────────────────────────
def build_result_df(
    point: np.ndarray,
    quantiles: np.ndarray,
    sku_index: list[str],
    last_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """将预测结果构建为 DataFrame。"""
    future_months = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    rows = []
    for i, sku in enumerate(sku_index):
        for h in range(horizon):
            rows.append(
                {
                    "sku_code": sku,
                    "forecast_month": future_months[h].strftime("%Y-%m"),
                    "forecast_qty": round(float(point[i, h]), 1),
                    "lower_80": round(float(quantiles[i, h, 1]), 1),  # q10 (index 1 in 2.5)
                    "upper_80": round(float(quantiles[i, h, 9]), 1),  # q90 (index 9 in 2.5)
                    "median": round(float(quantiles[i, h, 5]), 1),    # q50 (index 5 in 2.5)
                }
            )

    return pd.DataFrame(rows)


def save_json(
    point: np.ndarray,
    quantiles: np.ndarray,
    sku_index: list[str],
    last_date: pd.Timestamp,
    horizon: int,
    output_path: str,
) -> None:
    """输出为 JSON 格式 (按 SKU 分组)。"""
    future_months = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    month_labels = [m.strftime("%Y-%m") for m in future_months]

    results = {}
    for i, sku in enumerate(sku_index):
        results[sku] = {
            "months": month_labels,
            "forecast": [round(float(v), 1) for v in point[i]],
            "lower_80": [round(float(v), 1) for v in quantiles[i, :, 1]],
            "upper_80": [round(float(v), 1) for v in quantiles[i, :, 9]],
            "median": [round(float(v), 1) for v in quantiles[i, :, 5]],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ 已保存 {len(results)} 个 SKU 的预测到 {output_path}")


def save_csv(result_df: pd.DataFrame, output_path: str) -> None:
    """输出为 CSV 格式 (长格式)。"""
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存 {len(result_df)} 行预测到 {output_path}")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="九研 SKU 月度销量 × TimesFM 2.5 零样本预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python forecast_jiuyan.py                          # 全量预测
  python forecast_jiuyan.py --max-skus 50            # 快速试跑 50 个
  python forecast_jiuyan.py --horizon 6              # 预测 6 个月
  python forecast_jiuyan.py --skus 6974220210331     # 指定 SKU
  python forecast_jiuyan.py --format csv -o out.csv  # 输出 CSV
        """,
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help=f"预测步长/月 (默认 {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--min-months", type=int, default=DEFAULT_MIN_MONTHS,
        help=f"SKU 最低历史月数 (默认 {DEFAULT_MIN_MONTHS})",
    )
    parser.add_argument(
        "--skus", type=str, default=None,
        help="逗号分隔的 SKU 编码 (默认全部)",
    )
    parser.add_argument(
        "--max-skus", type=int, default=None,
        help="最大 SKU 数量 (快速试跑用)",
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help="SQLite 数据库路径",
    )
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="输出文件路径 (默认按最新销售流水日期命名: outputs/jiuyan_forecasts_YYYYMMDD.json)",
    )
    parser.add_argument(
        "--format", choices=["json", "csv"], default=None,
        help="输出格式 (默认从扩展名推断)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="模型推理批次大小 (默认 32)",
    )
    parser.add_argument(
        "--skip-check", action="store_true",
        help="跳过系统预检",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="启用日度补全修正 (针对当前未完结月份)",
    )
    parser.add_argument(
        "--daily-days", type=int, default=400,
        help="用于日度修正的背景天数 (默认 400 覆盖 1 年季节性)",
    )
    args = parser.parse_args()

    sku_filter = None
    if args.skus:
        sku_filter = [s.strip() for s in args.skus.split(",")]

    print("=" * 56)
    print("  九研 SKU 销量预测 — TimesFM 2.5 (PyTorch)")
    print("=" * 56)

    # 1. 加载数据
    print("\n📊 加载数据...")
    inputs, sku_index, last_date, latest_sale_ts, daily_contexts = load_data(
        db_path=args.db,
        min_months=args.min_months,
        sku_filter=sku_filter,
        max_skus=args.max_skus,
        refine=args.refine,
        daily_days=args.daily_days,
    )

    if args.output is None:
        args.output = str(build_default_output_path(output_dir, latest_sale_ts))

    # 推断输出格式
    out_format = args.format
    if not out_format:
        out_format = "csv" if Path(args.output).suffix.lower() == ".csv" else "json"

    # 2. 系统预检
    if not args.skip_check:
        print("\n🔍 系统预检...")
        max_ctx = max(len(s) for s in inputs)
        run_preflight(len(inputs), max_ctx, args.horizon)
    else:
        print("\n⚠️ 已跳过系统预检")

    # 3. 加载模型
    print("\n🤖 加载模型...")
    max_context = max(len(s) for s in inputs)
    # 对齐到 2 的幂次 (64 足以覆盖 60 个月)
    aligned_context = min(max(64, max_context), 16384)
    model = load_model(
        horizon=args.horizon,
        max_context=aligned_context,
        batch_size=args.batch_size,
    )

    # 4. 运行预测
    print(f"\n🚀 开始预测 (horizon={args.horizon}, SKU={len(inputs)})...")
    t_start = time.time()
    point, quantiles = run_forecast(model, inputs, args.horizon)
    point = np.maximum(point, 0)
    quantiles = np.maximum(quantiles, 0)
    t_total = time.time() - t_start
    print(f"   预测总耗时: {t_total:.1f}s ({t_total / len(inputs) * 1000:.0f} ms/SKU)")

    # ────────────── 命中当前月，执行日度修正 ──────────────
    refined_indices = set()
    if args.refine and latest_sale_ts is not None and not latest_sale_ts.is_month_end:
        print(f"\n✨ 正在执行日度修正 (Refinement)...")
        first_month_days = calendar.monthrange(latest_sale_ts.year, latest_sale_ts.month)[1]
        remaining_days = first_month_days - latest_sale_ts.day
        
        if remaining_days > 0:
            daily_point, daily_quantiles = run_forecast(model, daily_contexts, remaining_days)
            daily_point = np.maximum(daily_point, 0)
            daily_quantiles = np.maximum(daily_quantiles, 0)
            
            for i in range(len(sku_index)):
                actual_so_far = np.sum(daily_contexts[i][-latest_sale_ts.day:]) 
                predicted_remaining = np.sum(daily_point[i])
                refined_total = actual_so_far + predicted_remaining
                
                # 同步修正置信区间: 已发生的确切销量 + 剩余天数的区间预测积分
                remaining_lower80 = np.sum(daily_quantiles[i, :, 1])
                remaining_upper80 = np.sum(daily_quantiles[i, :, 9])
                remaining_median = np.sum(daily_quantiles[i, :, 5])
                
                point[i, 0] = refined_total
                quantiles[i, 0, 1] = actual_so_far + remaining_lower80
                quantiles[i, 0, 9] = actual_so_far + remaining_upper80
                quantiles[i, 0, 5] = actual_so_far + remaining_median

                refined_indices.add(i)
                if i < 3:
                     print(f"   SKU {sku_index[i]}: 首月预测修正为 {refined_total:.1f} = (实发:{actual_so_far:.0f} + 预测:{predicted_remaining:.1f})")
        else: # 刚好月末，理论上不会走到这里，因为前面拦截了
            for i in range(len(sku_index)):
                actual_sum = np.sum(daily_contexts[i][-latest_sale_ts.day:])
                point[i, 0] = actual_sum
                quantiles[i, 0, 1] = actual_sum
                quantiles[i, 0, 9] = actual_sum
                quantiles[i, 0, 5] = actual_sum
                refined_indices.add(i)

    # 5. 输出结果
    print(f"\n💾 保存结果 → {args.output}")
    if out_format == "csv":
        result_df = build_result_df(point, quantiles, sku_index, last_date, args.horizon)
        save_csv(result_df, args.output)
    else:
        save_json(point, quantiles, sku_index, last_date, args.horizon, args.output)

    # 6. 预览前 3 个 SKU
    print("\n── 前 3 个 SKU 预测预览 ──")
    future_months = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=args.horizon,
        freq="MS",
    )
    for i in range(min(3, len(sku_index))):
        sku = sku_index[i]
        historical_avg = float(np.mean(inputs[i][-12:]))
        print(f"\n  SKU: {sku} (近 12 月均值: {historical_avg:.0f})")
        print(f"  {'月份':>10s}  {'预测':>8s}  {'下限':>8s}  {'上限':>8s}")
        for h in range(min(6, args.horizon)):
            m_str = future_months[h].strftime('%Y-%m')
            # 标记是否被修正
            prefix_fmt = "(R) " + m_str if (h == 0 and i in refined_indices) else "  " + m_str
            print(
                f"{prefix_fmt:>12s}"
                f"  {point[i, h]:>8.0f}"
                f"  {quantiles[i, h, 1]:>8.0f}"
                f"  {quantiles[i, h, 9]:>8.0f}"
            )
        if args.horizon > 6:
            print(f"  ... 共 {args.horizon} 个月")

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
