#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path for TimesFM imports
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

try:
    import timesfm
except ImportError:
    print("🛑 Error: Cannot import timesfm. Ensure you are running with the correct virtualenv.")
    sys.exit(1)

# Configuration
DATA_PATH = "/Users/andychan/Documents/bigbang/raw_data/dlt/dlt_desc.csv"
OUTPUT_PATH = Path(__file__).parent / "dlt.csv"
HORIZON = 1
CONTEXT_LEN = 128  # Context length for TimesFM

def load_dlt_data(path):
    """Load DLT data and extract relevant columns (R1-R5, B1-B2)."""
    if not os.path.exists(path):
        print(f"🛑 Data file not found: {path}")
        sys.exit(1)
    
    # Headerless CSV, reverse rows to be chronological (top row is latest)
    df = pd.read_csv(path, header=None)
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Column mapping (DLT):
    # 0: Issue, 1: Date, 2-6: Red Balls (sorted), 7-8: Blue Balls
    red_balls = df.iloc[:, 2:7].values.astype(np.float32)
    blue_balls = df.iloc[:, 7:9].values.astype(np.float32)
    
    return red_balls, blue_balls

def forecast_balls(model, series_list):
    """Run forecast for each series in the list."""
    point, _ = model.forecast(inputs=series_list, freq=[0] * len(series_list))
    return point[:, 0]

def main():
    print("=" * 40)
    print(" DLT Lottery Forecast via TimesFM")
    print("=" * 40)
    
    # 1. Load Data
    print(f"📊 Loading data from {DATA_PATH}...")
    red_series, blue_series = load_dlt_data(DATA_PATH)
    
    # Combine into 7 series: R1-R5 + B1-B2
    inputs = []
    # Red balls R1-R5
    for i in range(5):
        inputs.append(red_series[:, i])
    # Blue balls B1-B2
    for i in range(2):
        inputs.append(blue_series[:, i])
    
    # 2. Init TimesFM Model
    print("🤖 Initializing TimesFM model...")
    hparams = timesfm.TimesFmHparams(
        context_len=CONTEXT_LEN,
        horizon_len=HORIZON,
        per_core_batch_size=32,
    )
    checkpoint = timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
    model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    
    # 3. Forecast
    print(f"🚀 Forecasting next set (7 numbers)...")
    raw_predictions = forecast_balls(model, inputs)
    
    # 4. Post-process (Rounding, Clipping, Uniqueness)
    # Red balls (1-35)
    red_pred = np.round(raw_predictions[:5]).astype(int)
    red_pred = np.clip(red_pred, 1, 35)
    
    # Ensure red balls are unique
    if len(set(red_pred)) < 5:
        print("⚠️ Handling duplicate red ball predictions...")
        unique_reds = []
        for r in red_pred:
            while r in unique_reds or r > 35 or r < 1:
                r = (r % 35) + 1
            unique_reds.append(r)
        red_pred = sorted(unique_reds)
    else:
        red_pred = sorted(red_pred)
    
    # Blue balls (1-12)
    blue_pred = np.round(raw_predictions[5:]).astype(int)
    blue_pred = np.clip(blue_pred, 1, 12)
    
    # Ensure blue balls are unique
    if len(set(blue_pred)) < 2:
        print("⚠️ Handling duplicate blue ball predictions...")
        unique_blues = []
        for b in blue_pred:
            while b in unique_blues or b > 12 or b < 1:
                b = (b % 12) + 1
            unique_blues.append(b)
        blue_pred = sorted(unique_blues)
    else:
        blue_pred = sorted(blue_pred)
    
    print(f"✨ Prediction: Red={red_pred}, Blue={blue_pred}")
    
    # 5. Save Output
    # Format: R1,R2,R3,R4,R5,B1,B2
    output_line = ",".join(map(str, red_pred)) + "," + ",".join(map(str, blue_pred)) + "\n"
    with open(OUTPUT_PATH, "w") as f:
        f.write(output_line)
    
    print(f"✅ Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
