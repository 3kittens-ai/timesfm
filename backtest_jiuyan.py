#!/usr/bin/env python3
"""
backtest_jiuyan.py — 九研 SKU 销量 TimesFM 纯净版回测

用截止到 2025-02 的数据预测 2025-03 到 2026-02 (未来 12 个月) 的销量，
并与数据库中 2025-03 到 2026-02 的实际销量进行对比，计算偏差 (MAE, MAPE, WAPE)。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = "/Users/andychan/Documents/jiuyan/sales_filtered_database/sales_filtered.sqlite"
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
    
    # 填充缺失月份为 0
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
        
        # 只保留未来刚好 horizon 长度的真实数据
        if len(future_df) < horizon:
            continue  # 没有完整实际数据无法全面对比
            
        future_series = future_df.iloc[:horizon]["monthly_qty"].values.astype(np.float32)
        history_series = history_df["monthly_qty"].values.astype(np.float32)
        
        if len(history_series) >= min_context_months:
            inputs.append(history_series)
            actuals.append(future_series)
            sku_index.append(sku)

    if max_skus and len(inputs) > max_skus:
        # 为了稳定，可以选择总销量排名靠前的 SKU 做测试
        # 这里为了简单直接取前 max_skus
        inputs = inputs[:max_skus]
        actuals = actuals[:max_skus]
        sku_index = sku_index[:max_skus]

    return inputs, actuals, sku_index

def load_model(horizon: int, max_context: int = 64, batch_size: int = 32):
    """加载 TimesFM 1.0 PyTorch 模型。"""
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    print("   正在加载 TimesFM 1.0 (200M)...")
    hparams = timesfm.TimesFmHparams(
        context_len=max_context,
        horizon_len=horizon,
        per_core_batch_size=batch_size,
    )
    checkpoint = timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
    model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    return model

def run_forecast(model, inputs: list[np.ndarray], horizon: int) -> np.ndarray:
    """批量预测"""
    all_point = []
    total = len(inputs)
    freqs = [0] * total  # 0 denotes monthly

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        batch = inputs[start:end]
        batch_freqs = freqs[start:end]
        print(f"   预测中: [{start + 1}~{end}] / {total} SKU ...", end="", flush=True)

        t0 = time.time()
        point, quantiles = model.forecast(inputs=batch, freq=batch_freqs)
        elapsed = time.time() - t0

        all_point.append(point)
        print(f" {elapsed:.1f}s")

    return np.concatenate(all_point, axis=0)

def main():
    parser = argparse.ArgumentParser(description="TimesFM 纯净版回测对比")
    parser.add_argument("--cutoff", type=str, default="2025-02-01", help="回测切割点 (默认 2025-02-01)")
    parser.add_argument("--horizon", type=int, default=12, help="预测步长 (默认 12)")
    parser.add_argument("--max-skus", type=int, default=0, help="限制测试 SKU 数 (0=所有合规)")
    parser.add_argument("--file", type=str, default="timesfm_backtest_results.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    print("=" * 60)
    print("  TimesFM 销量回测: 隐藏真实数据，预测并对比")
    print(f"  预测时间段: {args.cutoff} 之后 {args.horizon} 个月")
    print("=" * 60)

    max_skus_arg = args.max_skus if args.max_skus > 0 else None
    inputs, actuals, sku_index = load_backtest_data(
        DB_PATH, args.cutoff, args.horizon, min_context_months=12, max_skus=max_skus_arg
    )
    
    if len(inputs) == 0:
        print("🛑 找不到符合条件的 SKU。")
        sys.exit(1)

    print(f"\n📊 加载数据完成: 共 {len(inputs)} 个有效 SKU")
    
    max_context = min(max(64, max([len(i) for i in inputs])), 16384)
    model = load_model(args.horizon, max_context=max_context)

    print(f"\n🚀 开始纯净版预测...")
    point = run_forecast(model, inputs, args.horizon)
    actuals_np = np.array(actuals)
    
    # 将模型输出小于0的值归零
    point = np.maximum(point, 0)

    # ---------------- 整体误差统计 ----------------
    # 绝对误差 / 实际总量 (WAPE: Weighted Absolute Percentage Error)
    total_abs_error = np.sum(np.abs(point - actuals_np))
    total_actual = np.sum(actuals_np)
    wape = total_abs_error / total_actual if total_actual > 0 else 0
    
    mae = np.mean(np.abs(point - actuals_np))
    rmse = np.sqrt(np.mean((point - actuals_np) ** 2))
    
    print("\n" + "=" * 60)
    print("  回测结果总览 (全体 SKU 宏观评测)")
    print("=" * 60)
    print(f"  总真实销量: {total_actual:,.0f} 件")
    print(f"  总预测偏差: {total_abs_error:,.0f} 件")
    print(f"  全局 WAPE:  {wape * 100:.2f}% (总体偏差率，越低越好)")
    print(f"  平均绝对误差 (MAE):  {mae:.2f} 件/月/SKU")
    print(f"  均方根误差 (RMSE): {rmse:.2f} 件/月/SKU")

    # 构建汇总 DataFrame 供详情查阅
    rows = []
    future_months = pd.date_range(start=pd.Timestamp(args.cutoff) + pd.offsets.MonthBegin(1), periods=args.horizon, freq="MS")
    
    for i, sku in enumerate(sku_index):
        for h in range(args.horizon):
             rows.append({
                 "sku_code": sku,
                 "month": future_months[h].strftime("%Y-%m"),
                 "actual": float(actuals[i][h]),
                 "timesfm_point": round(float(point[i][h]), 1),
                 "abs_error": round(abs(float(point[i][h]) - float(actuals[i][h])), 1)
             })
             
    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.file, index=False, encoding="utf-8-sig")
    print(f"\n✅ 详细结果已导出至 {args.file}")
    
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
