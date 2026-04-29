#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import time
import calendar
import json
from pathlib import Path
from datetime import datetime, timedelta

# 配置
DB_PATH = "/Users/jiandang/Documents/Jiuyan/jiuyan-data/sales_filtered_database/sales_filtered.sqlite"
TARGET_SKUS_PATH = "/Users/jiandang/Documents/Jiuyan/timesfm/jiuyan-sales/outputs/backtest/target_skus.txt"
OUTPUT_PATH = "/Users/jiandang/Documents/Jiuyan/timesfm/jiuyan-sales/outputs/backtest/march_progress_results.json"
DAILY_CONTEXT_DAYS = 400
CHUNK_SIZE = 200

def load_skus(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_all_daily_sales(db_path, skus, start_date, end_date):
    """大批量获取日销量数据并缓存"""
    conn = sqlite3.connect(db_path)
    print(f"   📡 正在从数据库批量抓取日销量 ({start_date} ~ {end_date})...")
    query = f"""
        SELECT barcode as sku, sale_date, SUM(sales_volume) as daily_vol
        FROM sales
        WHERE sale_date BETWEEN '{start_date}' AND '{end_date}'
        AND barcode IN ({','.join(['?']*len(skus))})
        GROUP BY barcode, sale_date
    """
    df = pd.read_sql(query, conn, params=skus)
    conn.close()
    
    # 转换为 dict: {sku: {date_str: vol}}
    cache = {}
    for sku, group in df.groupby("sku"):
        cache[sku] = dict(zip(group["sale_date"], group["daily_vol"]))
    return cache

def get_daily_series(cache, sku, end_date, days):
    """从缓存中提取指定长度的序列"""
    end_ts = pd.Timestamp(end_date)
    date_range = [(end_ts - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)][::-1]
    
    sku_cache = cache.get(sku, {})
    return np.array([sku_cache.get(d, 0) for d in date_range], dtype=np.float32)

def get_monthly_inputs(db_path, skus, cutoff_month):
    """获取截止月及之前的月度销量序列"""
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT variant_key as sku, year_month, monthly_qty
        FROM v_training_base
        WHERE variant_key IN ({','.join(['?']*len(skus))})
    """
    df = pd.read_sql(query, conn, params=skus)
    conn.close()
    
    cutoff_ts = pd.Timestamp(cutoff_month + "-01")
    results = {}
    for sku, group in df.groupby("sku"):
        group["date"] = pd.to_datetime(group["year_month"] + "-01")
        group = group.sort_values("date")
        history = group[group["date"] <= cutoff_ts]
        results[sku] = history["monthly_qty"].values.astype(np.float32)
        
    for sku in skus:
        if sku not in results:
            results[sku] = np.array([0], dtype=np.float32)
            
    return results

def load_timesfm_model():
    """加载并编译 TimesFM"""
    import torch
    import timesfm
    torch.set_float32_matmul_precision("high")
    print("   正在加载 TimesFM 2.5 (200M)...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    print("   编译模型...")
    model.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=256, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True,
        infer_is_positive=True, fix_quantile_crossing=True,
    ))
    return model

def calculate_metrics(actual_totals, forecast_totals):
    actual_totals = np.array(actual_totals)
    forecast_totals = np.array(forecast_totals)
    
    abs_err = np.abs(actual_totals - forecast_totals)
    sum_abs_err = np.sum(abs_err)
    sum_actual = np.sum(actual_totals)
    
    wape = sum_abs_err / sum_actual if sum_actual > 0 else 0
    mae = np.mean(abs_err)
    
    return float(wape), float(mae)

def main():
    skus = load_skus(TARGET_SKUS_PATH)
    print(f"📊 加载了 {len(skus)} 个 SKU。")
    
    # 提前缓存所有需要的日数据 (2025年至今)
    # 起始日期: 2026-03-01 往回走 400 天 = 2025-01-25 左右
    daily_cache = get_all_daily_sales(DB_PATH, skus, "2025-01-01", "2026-03-31")
    
    model = load_timesfm_model()
    
    # 1. 准备月度基准模型输入 (截止到 2026-02)
    print("📅 准备月度基准序列...")
    monthly_inputs_list = get_monthly_inputs(DB_PATH, skus, "2026-02")
    inputs_list = [monthly_inputs_list.get(s, np.array([0], dtype=np.float32)) for s in skus]
    
    # 2. 跑月度基准预测
    print("🚀 运行 3 月月度基准预测...")
    point_fc_monthly = []
    for s in range(0, len(inputs_list), CHUNK_SIZE):
        batch = inputs_list[s:s+CHUNK_SIZE]
        p, _ = model.forecast(horizon=1, inputs=batch)
        point_fc_monthly.append(np.maximum(p[:, 0], 0))
    baseline_forecasts = np.concatenate(point_fc_monthly)
    
    # 获取 2026-03 的真实全月销量
    print("✅ 计算 2026-03 真实总销量...")
    march_dates = [f"2026-03-{d:02d}" for d in range(1, 32)]
    actual_march_totals = []
    for sku in skus:
        sku_cache = daily_cache.get(sku, {})
        total = sum([sku_cache.get(d, 0) for d in march_dates])
        actual_march_totals.append(total)
    
    # 计算基准指标
    base_wape, base_mae = calculate_metrics(actual_march_totals, baseline_forecasts)
    print(f"📉 基准预测 (仅月度): WAPE = {base_wape*100:.2f}%, MAE = {base_mae:.2f}")
    
    progress_results = [{
        "day": 0,
        "label": "Baseline (Monthly only)",
        "wape": base_wape,
        "mae": base_mae
    }]
    
    # 3. 循环 3 月的 1 到 31 日，执行修正
    for day in range(1, 32):
        cutoff_date = f"2026-03-{day:02d}"
        print(f"🕒 模拟截止日期: {cutoff_date} ... ", end="", flush=True)
        
        # 从缓存提取日度序列 (400天)
        daily_inputs = [get_daily_series(daily_cache, s, cutoff_date, DAILY_CONTEXT_DAYS) for s in skus]
        
        # 预测本月剩余天数
        remaining_days = 31 - day
        refined_forecasts = []
        
        if remaining_days > 0:
            daily_p_list = []
            for s in range(0, len(daily_inputs), CHUNK_SIZE):
                batch = daily_inputs[s:s+CHUNK_SIZE]
                p, _ = model.forecast(horizon=remaining_days, inputs=batch)
                daily_p_list.append(p)
            daily_point = np.maximum(np.concatenate(daily_p_list, axis=0), 0)
            
            for i in range(len(skus)):
                # 已发
                actual_so_far = np.sum(daily_inputs[i][-day:])
                pred_remaining = np.sum(daily_point[i])
                refined_forecasts.append(actual_so_far + pred_remaining)
        else:
            # 31号
            for i in range(len(skus)):
                refined_forecasts.append(np.sum(daily_inputs[i][-31:]))
        
        wape, mae = calculate_metrics(actual_march_totals, refined_forecasts)
        progress_results.append({
            "day": day,
            "label": f"March {day:02d}",
            "wape": wape,
            "mae": mae
        })
        print(f"WAPE: {wape*100:.2f}%")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(progress_results, f, indent=2)
    print(f"✅ 结果已保存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
