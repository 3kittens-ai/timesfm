import pandas as pd
import sqlite3
from pathlib import Path

# Paths
RESULTS_PATH = Path("outputs/backtest/timesfm_backtest_2.5_xreg_v4_results.csv")
PIVOT_PATH = Path("outputs/timesfm_forecast_12m_pivot_xreg_v4.csv")
DB_PATH = Path("/Users/jiandang/Documents/Jiuyan/jiuyan-data/sales_filtered_database/sales_filtered.sqlite")

def update_pivot():
    print(f"Loading results from {RESULTS_PATH}...")
    df_results = pd.read_csv(RESULTS_PATH)
    
    # Check for empty data
    if df_results.empty:
        print("Error: Results file is empty.")
        return

    # Pivot to monthly columns
    print("Pivoting data...")
    # Convert 'month' format if needed
    df_results['month_short'] = df_results['month'].apply(lambda x: x[:7]) # YYYY-MM
    pivot = df_results.pivot(index='sku_code', columns='month_short', values='forecast_v4')
    
    # Calculate 12-month total
    month_cols = sorted(pivot.columns.tolist())
    pivot['12个月合计'] = pivot[month_cols].sum(axis=1).round(1)
    
    # Fetch SKU metadata from DB
    print("Fetching SKU metadata from database...")
    conn = sqlite3.connect(DB_PATH)
    sku_query = "SELECT DISTINCT sku_code, sku_name, family_tags FROM dim_sku"
    df_meta = pd.read_sql(sku_query, conn)
    conn.close()
    
    # Merge
    print("Merging results with metadata...")
    df_meta['sku_code'] = df_meta['sku_code'].astype(str)
    pivot_df = pivot.reset_index()
    pivot_df['sku_code'] = pivot_df['sku_code'].astype(str)
    final = df_meta.merge(pivot_df, on='sku_code', how='inner')
    
    # Organize columns
    # [sku_code, sku_name] + month_cols + [12个月合计, family_tags]
    cols = ['sku_code', 'sku_name'] + month_cols + ['12个月合计', 'family_tags']
    final = final[cols]
    
    # Rename family_tags to 家族标签 to match user's previous format if possible
    # (Actually, in the head output it was '家族标签')
    final = final.rename(columns={'family_tags': '家族标签'})
    
    print(f"Saving updated pivot to {PIVOT_PATH}...")
    final.to_csv(PIVOT_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    update_pivot()
