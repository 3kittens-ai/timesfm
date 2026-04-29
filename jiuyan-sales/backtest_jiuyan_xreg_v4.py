#!/usr/bin/env python3
"""
backtest_jiuyan_xreg_v4.py — 引入进阶外部变量 (XReg) 的九研销量回测
包含：日均动量 (Velocity)、去年同比节奏 (YoY Step)、节日偏移 (Holiday Offsets)
针对 2GB RAM 进行了极致优化。
"""

import argparse
import sqlite3
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = "/Users/jiandang/Documents/Jiuyan/jiuyan-data/sales_filtered_database/sales_filtered.sqlite"
CHUNK_SIZE = 50 

FEATURE_SETS = {
    "full": {
        "static_cat": ["category", "family"],
        "static_num": ["price", "v7", "v15"],
        "dynamic_num": ["cny_offset", "is_618", "is_double11", "is_double12", "yoy_step"],
    },
    "v15": {
        "static_cat": [],
        "static_num": ["v15"],
        "dynamic_num": [],
    },
    "v7": {
        "static_cat": [],
        "static_num": ["v7"],
        "dynamic_num": [],
    },
    "v7_v15": {
        "static_cat": [],
        "static_num": ["v7", "v15"],
        "dynamic_num": [],
    },
    "yoy_step": {
        "static_cat": [],
        "static_num": [],
        "dynamic_num": ["yoy_step"],
    },
}

# 节日日期定义 (用于计算 Month Offset)
CNY_MONTHS = {2023: 1, 2024: 2, 2025: 1, 2026: 2}
MID_AUTUMN_MONTHS = {2023: 9, 2024: 9, 2025: 10, 2026: 9}

def get_holiday_offset(year, month, holiday_dict):
    if year in holiday_dict:
        target_month = holiday_dict[year]
        return month - target_month
    return 0

def load_v4_data(
    db_path: str,
    cutoff_date: str,
    horizon: int,
    max_skus: int | None = None,
    sku_file: str | None = None,
):
    conn = sqlite3.connect(db_path)
    cutoff_ts = pd.Timestamp(cutoff_date)
    
    # 1. 获取选定的 SKU 列表
    if sku_file and Path(sku_file).exists():
        print(f"   [1/3] 从 {sku_file} 加载 SKU 列表...")
        with open(sku_file, "r") as f:
            target_skus = [line.strip() for line in f if line.strip()]
        if max_skus:
            target_skus = target_skus[:max_skus]
        print(f"         加载了 {len(target_skus)} 个 SKU。")
    else:
        print(f"   [1/3] 确定目标 SKU (从数据库限制前 {max_skus} 个)...")
        sku_query = f"SELECT DISTINCT variant_key FROM v_training_base LIMIT {max_skus if max_skus else 1000000}"
        target_skus = pd.read_sql(sku_query, conn)["variant_key"].tolist()
    
    sku_placeholders = ",".join(["?"] * len(target_skus))
    query_base = f"""
    SELECT 
        s.variant_key, s.year_month, s.monthly_qty,
        d.product_category, d.family_tags, d.last_tag_price, d.sku_code
    FROM v_training_base s
    LEFT JOIN dim_sku d ON s.variant_key = d.variant_key
    WHERE s.variant_key IN ({sku_placeholders})
    """
    df = pd.read_sql(query_base, conn, params=target_skus)
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    
    # 2. 获取日销量动能 (Velocity)
    print("   [2/3] 从原始交易表提取近期动能 (7/15天)...")
    v7_end = cutoff_ts
    v7_start = v7_end - pd.Timedelta(days=6)
    v15_start = v7_end - pd.Timedelta(days=14)
    
    query_velocity = f"""
    SELECT barcode as sku_code, 
           SUM(CASE WHEN sale_date >= '{v7_start.strftime('%Y-%m-%d')}' THEN sales_volume ELSE 0 END) / 7.0 as v7,
           SUM(CASE WHEN sale_date >= '{v15_start.strftime('%Y-%m-%d')}' THEN sales_volume ELSE 0 END) / 15.0 as v15
    FROM sales
    WHERE sale_date BETWEEN '{v15_start.strftime('%Y-%m-%d')}' AND '{v7_end.strftime('%Y-%m-%d')}'
    GROUP BY barcode
    """
    velocity_df = pd.read_sql(query_velocity, conn)
    
    # 3. 数据拆分与循环处理
    print("   [3/3] 执行特征工程与时序对齐...")
    global_max = df["date"].max()
    all_data = []
    
    # 合并动量到主表暂存以加速 (通过 sku_code)
    # Note: dim_sku.sku_code matches sales.barcode
    
    for sku, group in df.groupby("variant_key"):
        group = group.sort_values("date")
        if group["date"].min() > cutoff_ts: continue
            
        horizon_end = cutoff_ts + pd.DateOffset(months=horizon)
        full_range = pd.date_range(start=group["date"].min(), end=horizon_end, freq="MS")
        merged = pd.DataFrame({"date": full_range}).merge(group, on="date", how="left")
        observed_mask = merged["monthly_qty"].notna()
        merged["monthly_qty"] = merged["monthly_qty"].fillna(0)
        
        static_info = group.iloc[0]
        sku_code = static_info["sku_code"]
        
        # 准备历史
        history_df = merged[merged["date"] <= cutoff_ts]
        if len(history_df) < 12:
            continue
        
        history_series = history_df["monthly_qty"].values.astype(np.float32)
        future_df = merged[merged["date"] > cutoff_ts].copy()
        future_observed = observed_mask[merged["date"] > cutoff_ts].to_numpy()
        actual_series = np.full(horizon, np.nan, dtype=np.float32)
        actual_len = min(len(future_df), horizon)
        if actual_len > 0:
            actual_values = future_df.iloc[:actual_len]["monthly_qty"].values.astype(np.float32)
            actual_series[:actual_len] = np.where(
                future_observed[:actual_len],
                actual_values,
                np.nan,
            )

        # A. 动态特征生成 (全周期: 历史 + 未来)
        merged["year"] = merged["date"].dt.year
        merged["month"] = merged["date"].dt.month
        
        # 1. 去年同比 MoM 节奏 (YoY Step)
        # 获取去年的销量序列平移 12 个月。为简化，直接算历史中该月的同比环比。
        merged["lag_12"] = merged["monthly_qty"].shift(12)
        merged["lag_13"] = merged["monthly_qty"].shift(13)
        # Step = 去年当月 / 去年上月
        merged["yoy_step"] = (merged["lag_12"] / merged["lag_13"].replace(0, np.nan)).fillna(1.0).clip(0.1, 10.0)
        
        # 2. 春节偏移（数值特征，避免 one-hot 在未来月份产生大量 unseen categories）
        merged["cny_offset"] = merged.apply(lambda x: get_holiday_offset(x["year"], x["month"], CNY_MONTHS), axis=1)
        
        # 3. 大促特征拆分为具体活动月份，避免把不同促销强度混成一个类别
        merged["is_618"] = (merged["month"] == 6).astype(np.float32)
        merged["is_double11"] = (merged["month"] == 11).astype(np.float32)
        merged["is_double12"] = (merged["month"] == 12).astype(np.float32)
        
        # B. 动能特征 (从 velocity_df 取)
        v_row = velocity_df[velocity_df["sku_code"] == sku_code]
        v7 = float(v_row["v7"].values[0]) if len(v_row) > 0 else 0
        v15 = float(v_row["v15"].values[0]) if len(v_row) > 0 else 0

        all_data.append({
            "sku": sku,
            "inputs": history_series,
            "actuals": actual_series,
            "static_cat": {
                "category": str(static_info["product_category"] or "Other"),
                "family": str(static_info["family_tags"] or "None").split("|")[0]
            },
            "static_num": {
                "price": float(static_info["last_tag_price"] or 0),
                "v7": v7,
                "v15": v15
            },
            'dynamic_num': {
                'cny_offset': merged['cny_offset'].values.astype(np.float32).tolist(),
                'is_618': merged['is_618'].values.tolist(),
                'is_double11': merged['is_double11'].values.tolist(),
                'is_double12': merged['is_double12'].values.tolist(),
                'yoy_step': merged['yoy_step'].values.tolist(),
            },
            'dynamic_cat': {}
        })
        if max_skus and len(all_data) >= max_skus: break

    conn.close()
    print(f"         保留 {len(all_data)} 个 SKU。")
    return all_data

def run_v4_forecast(model, data_list, horizon, feature_set: str):
    import torch
    all_point = []
    total = len(data_list)
    feature_config = FEATURE_SETS[feature_set]
    
    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        batch = data_list[start:end]
        inputs = [d["inputs"] for d in batch]

        print(f"   [V4-XReg] 预测进度: [{start + 1}~{end}] / {total} ...", end="", flush=True)
        t0 = time.time()
        s_cat = (
            {k: [d["static_cat"][k] for d in batch] for k in feature_config["static_cat"]}
            if feature_config["static_cat"]
            else None
        )
        s_num = (
            {k: [d["static_num"][k] for d in batch] for k in feature_config["static_num"]}
            if feature_config["static_num"]
            else None
        )
        d_num = (
            {k: [d["dynamic_num"][k] for d in batch] for k in feature_config["dynamic_num"]}
            if feature_config["dynamic_num"]
            else None
        )

        point, _ = model.forecast_with_covariates(
            inputs=inputs,
            static_categorical_covariates=s_cat,
            static_numerical_covariates=s_num,
            dynamic_numerical_covariates=d_num,
            xreg_mode="timesfm + xreg"
        )

        print(f" {time.time() - t0:.1f}s")
        all_point.extend(np.asarray(pred, dtype=np.float32)[:horizon] for pred in point)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return np.stack(all_point, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=str, default="2025-02-01")
    parser.add_argument("--max-skus", type=int, default=0) # 0 表示不限制，除非提供了 sku-file
    parser.add_argument("--sku-file", type=str, default=None)
    parser.add_argument("--feature-set", type=str, default="full", choices=sorted(FEATURE_SETS.keys()))
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    import timesfm

    data_list = load_v4_data(DB_PATH, args.cutoff, 12, max_skus=args.max_skus if args.max_skus > 0 else None, sku_file=args.sku_file)
    if not data_list: return
    
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=256, normalize_inputs=True,
        per_core_batch_size=32, return_backcast=True
    ))

    point = run_v4_forecast(model, data_list, 12, feature_set=args.feature_set)
    point = np.maximum(point, 0)
    
    default_file = (
        Path("/Users/jiandang/Documents/Jiuyan/timesfm/jiuyan-sales/outputs/backtest")
        / f"timesfm_backtest_2.5_xreg_v4_{args.feature_set}_results.csv"
    )
    out_file = Path(args.file) if args.file else default_file
    results = []
    import datetime
    cutoff_dt = datetime.datetime.strptime(args.cutoff, "%Y-%m-%d")
    
    for i, d in enumerate(data_list):
        for h in range(12):
            # Calculate the year_month for the forecast point
            month_dt = cutoff_dt + pd.DateOffset(months=h+1)
            actual_val = float(d["actuals"][h]) if h < len(d["actuals"]) else float("nan")
            results.append({
                "sku_code": d["sku"],
                "month": month_dt.strftime("%Y-%m-%d"),
                "actual": actual_val if not np.isnan(actual_val) else None,
                "forecast_v4": round(float(point[i][h]), 1)
            })
    
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n✨ V4 回测成功！结果：{out_file}")

if __name__ == "__main__":
    main()
