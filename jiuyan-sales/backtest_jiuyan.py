#!/usr/bin/env python3
"""
backtest_jiuyan.py — 九研 SKU 销量 TimesFM 2.5 纯净版回测

用截止到 2025-02 的数据预测 2025-03 到 2026-02 (未来 12 个月) 的销量，
并与数据库中 2025-03 到 2026-02 的实际销量进行对比，计算偏差 (MAE, MAPE, WAPE)。
支持输出明细 CSV，并可选输出 JSON 和 Markdown 汇总报表。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = "/Users/jiandang/Documents/Jiuyan/jiuyan-data/sales_filtered_database/sales_filtered.sqlite"
CHUNK_SIZE = 200

def load_backtest_data(
    db_path: str, cutoff_date: str, horizon: int, min_context_months: int = 12, max_skus: int | None = None
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    加载回测数据。
    返回:
      inputs: 历史输入数据 (cutoff_date 及之前)
      actuals: 真实对齐的未来数据 (cutoff_date 之后，长度为 horizon)
      sku_index: SKU 列表
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT variant_key, year_month, monthly_qty FROM v_training_base", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    cutoff_ts = pd.Timestamp(cutoff_date)
    
    # 获取全局最大日期
    global_max = df["date"].max()
    
    inputs = []
    actuals = []
    sku_index = []

    for sku, group in df.groupby("variant_key"):
        group = group.sort_values("date")
        start = group["date"].min()
        
        # 如果这个SKU在截止日期前没出现过，跳过
        if start > cutoff_ts:
            continue
            
        full_range = pd.date_range(start=start, end=global_max, freq="MS")
        full_df = pd.DataFrame({"date": full_range})
        merged = full_df.merge(group[["date", "monthly_qty"]], on="date", how="left")
        merged["monthly_qty"] = merged["monthly_qty"].fillna(0)
        
        # 拆分历史和未来
        history_df = merged[merged["date"] <= cutoff_ts]
        future_df = merged[merged["date"] > cutoff_ts]
        
        # 允许未来数据长度小于 horizon (用于预测未来)
        future_series = np.full(horizon, np.nan, dtype=np.float32)
        actual_len = min(len(future_df), horizon)
        if actual_len > 0:
            future_series[:actual_len] = future_df.iloc[:actual_len]["monthly_qty"].values.astype(np.float32)
        
        history_series = history_df["monthly_qty"].values.astype(np.float32)
        
        if len(history_series) >= min_context_months:
            inputs.append(history_series)
            actuals.append(future_series)
            sku_index.append(sku)

    if max_skus and len(inputs) > max_skus:
        inputs = inputs[:max_skus]
        actuals = actuals[:max_skus]
        sku_index = sku_index[:max_skus]

    return inputs, actuals, sku_index

def load_model(horizon: int, max_context: int = 1024, batch_size: int = 32):
    """加载 TimesFM 2.5 PyTorch 模型并编译。"""
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    print(f"   正在加载 TimesFM 2.5 (200M)...")
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

def run_forecast(model, inputs: list[np.ndarray], horizon: int) -> np.ndarray:
    """批量预测"""
    all_point = []
    total = len(inputs)

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        batch = inputs[start:end]
        print(f"   预测中: [{start + 1}~{end}] / {total} SKU ...", end="", flush=True)

        t0 = time.time()
        point, quantiles = model.forecast(horizon=horizon, inputs=batch)
        elapsed = time.time() - t0

        all_point.append(point)
        print(f" {elapsed:.1f}s")

    return np.concatenate(all_point, axis=0)

def main():
    parser = argparse.ArgumentParser(description="TimesFM 2.5 销量回测对比")
    output_dir = Path(__file__).parent / "outputs" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    default_csv = output_dir / "timesfm_backtest_2.5_results.csv"
    default_json = output_dir / "timesfm_backtest_2.5_results.json"
    default_md = output_dir / "timesfm_backtest_2.5_summary.md"

    parser.add_argument("--cutoff", type=str, default="2025-02-01", help="回测切割点 (默认 2025-02-01)")
    parser.add_argument("--horizon", type=int, default=12, help="预测步长 (默认 12)")
    parser.add_argument("--max-skus", type=int, default=0, help="限制测试 SKU 数 (0=所有合规)")
    parser.add_argument("--skus", type=str, default="", help="指定特定的 SKU 编码 (逗号分隔或文件路径)")
    parser.add_argument("--file", type=str, default=str(default_csv), help=f"输出 CSV 路径 (默认 {default_csv})")
    parser.add_argument("--json", type=str, default=str(default_json), help=f"输出 JSON 指标路径 (默认 {default_json})")
    parser.add_argument("--md", type=str, default=str(default_md), help=f"输出 Markdown 报告路径 (默认 {default_md})")
    args = parser.parse_args()

    print("=" * 60)
    print("  TimesFM 2.5 销量回测: 隐藏真实数据，预测并对比")
    print(f"  预测时间段: {args.cutoff} 之后 {args.horizon} 个月")
    print("=" * 60)

    # 处理特定的 SKUs
    target_skus = []
    if args.skus:
        if Path(args.skus).is_file():
            with open(args.skus, "r") as f:
                target_skus = [line.strip() for line in f if line.strip()]
        else:
            target_skus = [s.strip() for s in args.skus.split(",")]
        args.max_skus = 0 

    max_skus_arg = args.max_skus if args.max_skus > 0 else None
    inputs, actuals, sku_index = load_backtest_data(
        DB_PATH, args.cutoff, args.horizon, min_context_months=12, max_skus=max_skus_arg
    )
    
    # 手动过滤指定 SKU
    if target_skus:
        filtered_inputs, filtered_actuals, filtered_sku_index = [], [], []
        target_set = set(target_skus)
        for i, sku in enumerate(sku_index):
            clean_sku = str(sku).split(".")[0]
            if clean_sku in target_set or str(sku) in target_set:
                filtered_inputs.append(inputs[i])
                filtered_actuals.append(actuals[i])
                filtered_sku_index.append(sku)
        inputs, actuals, sku_index = filtered_inputs, filtered_actuals, filtered_sku_index

    if len(inputs) == 0:
        print("🛑 找不到符合条件的 SKU。")
        sys.exit(1)

    print(f"\n📊 加载数据完成: 共 {len(inputs)} 个有效 SKU")
    
    max_context = min(max(64, max([len(i) for i in inputs])), 16384)
    model = load_model(args.horizon, max_context=max_context)

    print(f"\n🚀 开始 TimesFM 2.5 纯净版预测...")
    point = run_forecast(model, inputs, args.horizon)
    actuals_np = np.array(actuals)
    point = np.maximum(point, 0)

    # ---------------- 整体误差统计 (只统计非 NaN 的月份) ----------------
    mask = ~np.isnan(actuals_np)
    if np.any(mask):
        valid_point = point[mask]
        valid_actual = actuals_np[mask]
        total_abs_error = np.sum(np.abs(valid_point - valid_actual))
        total_actual = np.sum(valid_actual)
        wape = total_abs_error / total_actual if total_actual > 0 else 0
        mae = np.mean(np.abs(valid_point - valid_actual))
        rmse = np.sqrt(np.mean((valid_point - valid_actual) ** 2))
    else:
        total_abs_error = 0
        total_actual = 0
        wape = 0
        mae = 0
        rmse = 0
    
    print("\n" + "=" * 60)
    print("  回测结果总览 (只统计已有真实数据的月份，如 2026-03)")
    print("=" * 60)
    print(f"  统计样本数 (SKU-月): {np.sum(mask)}")
    print(f"  总真实销量: {total_actual:,.0f} 件")
    print(f"  总预测偏差: {total_abs_error:,.0f} 件")
    print(f"  全局 WAPE:  {wape * 100:.2f}%")
    print(f"  平均绝对误差 (MAE):  {mae:.2f} 件/月/SKU")

    # 构建详情与汇总
    rows = []
    summary_rows = []
    future_months = pd.date_range(start=pd.Timestamp(args.cutoff) + pd.offsets.MonthBegin(1), periods=args.horizon, freq="MS")
    
    for i, sku in enumerate(sku_index):
        for h in range(args.horizon):
             act_val = float(actuals[i][h])
             rows.append({
                 "sku_code": sku,
                 "month": future_months[h].strftime("%Y-%m"),
                 "actual": act_val if not np.isnan(act_val) else None,
                 "forecast": round(float(point[i][h]), 1),
             })
        
        sku_mask = ~np.isnan(actuals[i])
        if np.any(sku_mask):
            sum_act = np.sum(actuals[i][sku_mask])
            sum_pred = np.sum(point[i][sku_mask])
            sum_abs_err = np.sum(np.abs(point[i][sku_mask] - actuals[i][sku_mask]))
            sku_wape = sum_abs_err / sum_act if sum_act > 0 else 0
            sku_bias = (sum_pred - sum_act) / sum_act if sum_act > 0 else 0
        else:
            sum_act, sum_pred, sum_abs_err, sku_wape, sku_bias = 0, 0, 0, 0, 0

        summary_rows.append({
            "sku_code": sku,
            "total_actual": int(sum_act),
            "total_forecast": int(sum_pred),
            "abs_error": int(sum_abs_err),
            "wape": float(sku_wape),
            "bias": float(sku_bias)
        })
             
    result_df = pd.DataFrame(rows, columns=["sku_code", "month", "actual", "forecast"])
    result_df.to_csv(args.file, index=False, encoding="utf-8-sig")
    print(f"\n✅ 详细结果已导出至 {args.file}")

    if args.json:
        json_data = {
            "metrics": {
                "total_actual": int(total_actual),
                "total_abs_error": int(total_abs_error),
                "wape": float(wape),
                "mae": float(mae),
                "rmse": float(rmse)
            },
            "skus": summary_rows
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 指标数据已导出至 {args.json}")

    if args.md:
        md_content = [
            f"# TimesFM 2.5 回测报告 ({args.cutoff}, {args.horizon}个月)",
            "",
            "## 1. 宏观指标概览",
            f"- **总真实销量**: {total_actual:,.0f}",
            f"- **总预测偏差**: {total_abs_error:,.0f}",
            f"- **全局 WAPE**: {wape * 100:.2f}%",
            f"- **MAE**: {mae:.2f}",
            "",
            "## 2. SKU 详情对照表",
            "| SKU 编码 | 实际总销量 | 预测总销量 | 绝对误差量 | WAPE | Bias |",
            "|---|---:|---:|---:|---:|---:|"
        ]
        sorted_summary = sorted(summary_rows, key=lambda x: x["total_actual"], reverse=True)
        for s in sorted_summary:
            wape_str = f"{s['wape']*100:.1f}%"
            bias_str = f"{s['bias']*100:+.1f}%"
            md_content.append(f"| {s['sku_code']} | {s['total_actual']:,} | {s['total_forecast']:,} | {s['abs_error']:,} | {wape_str} | {bias_str} |")
        
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
        print(f"✅ 回测报告已生成至 {args.md}")
    
    print("\n── 随机抽样 3 个 SKU 全周期对比 ──")
    sample_indices = np.random.choice(len(sku_index), min(3, len(sku_index)), replace=False)
    for i in sample_indices:
        sku = sku_index[i]
        sum_act = np.sum(actuals[i])
        sum_pred = np.sum(point[i])
        diff_perc = ((sum_pred - sum_act) / sum_act * 100) if sum_act > 0 else 0
        print(f"\n  SKU: {sku}")
        print(f"  全年实际总销: {sum_act:.0f}  |  全年预测总销: {sum_pred:.0f}  |  总偏差: {diff_perc:+.1f}%")
        print(f"  {'月份':>10s}  {'实际':>8s}  {'预测':>8s}  {'误差':>8s}")
        for h in range(args.horizon):
            act = actuals[i][h]
            pred = point[i][h]
            err = pred - act
            print(f"  {future_months[h].strftime('%Y-%m'):>10s}  {act:>8.0f}  {pred:>8.0f}  {err:>+8.0f}")

if __name__ == "__main__":
    main()
