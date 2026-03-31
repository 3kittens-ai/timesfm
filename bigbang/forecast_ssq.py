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
DATA_PATH = "/Users/andychan/Documents/bigbang/raw_data/ssq/ssq_desc.csv"
OUTPUT_PATH = Path(__file__).parent / "ssq.csv"
HORIZON = 1
CONTEXT_LEN = 128  # Context length for TimesFM

def load_ssq_data(path):
    """Load SSQ data and extract relevant columns (R1-R6, Blue)."""
    if not os.path.exists(path):
        print(f"🛑 Data file not found: {path}")
        sys.exit(1)
    
    # Headerless CSV, reverse rows to be chronological (top row is latest)
    df = pd.read_csv(path, header=None)
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Column mapping:
    # 0: Issue, 1: Date, 2-7: Red Balls (sorted), 8: Blue
    red_balls = df.iloc[:, 2:8].values.astype(np.float32)
    blue_ball = df.iloc[:, 8:9].values.astype(np.float32)
    
    return red_balls, blue_ball

def forecast_balls(model, series_list):
    """Run forecast for each series in the list."""
    # TimesFM model.forecast expects list of 1D arrays
    # Preparing data: Transpose to have (7, N) where each row is a time series
    point, _ = model.forecast(inputs=series_list, freq=[0] * len(series_list))
    return point[:, 0] # Return the first step of forecast for each series

def main():
    print("=" * 40)
    print(" SSQ Lottery Forecast via TimesFM")
    print("=" * 40)
    
    # 1. Load Data
    print(f"📊 Loading data from {DATA_PATH}...")
    red_series, blue_series = load_ssq_data(DATA_PATH)
    
    # Combine into 7 series: R1-R6 + Blue
    inputs = []
    # Red balls R1-R6
    for i in range(6):
        inputs.append(red_series[:, i])
    # Blue ball
    inputs.append(blue_series[:, 0])
    
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
    # Red balls (1-33)
    red_pred = np.round(raw_predictions[:6]).astype(int)
    red_pred = np.clip(red_pred, 1, 33)
    
    # Ensure red balls are unique
    if len(set(red_pred)) < 6:
        print("⚠️ Handling duplicate red ball predictions...")
        # Simple adjustment: shift duplicates until unique
        unique_reds = []
        for r in red_pred:
            while r in unique_reds or r > 33 or r < 1:
                r = (r % 33) + 1
            unique_reds.append(r)
        red_pred = sorted(unique_reds)
    else:
        red_pred = sorted(red_pred)
    
    # Blue ball (1-16)
    blue_pred = int(np.round(raw_predictions[6]))
    blue_pred = max(1, min(16, blue_pred))
    
    print(f"✨ Prediction: Red={red_pred}, Blue={blue_pred}")
    
    # 5. Save Output
    # Format: R1,R2,R3,R4,R5,R6,Blue
    output_line = ",".join(map(str, red_pred)) + f",{blue_pred}\n"
    with open(OUTPUT_PATH, "w") as f:
        f.write(output_line)
    
    print(f"✅ Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
