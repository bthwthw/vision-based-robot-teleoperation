import sys
import os
import numpy as np
import pandas as pd
 
def estimate_lag(t, raw, filt):
    """
    Estimate the lag (in seconds) between raw and filt using cross-correlation.
    Return a positive lag value indicating that filt lags behind raw by that many seconds.
    """
    raw = np.asarray(raw) - np.mean(raw)
    filt = np.asarray(filt) - np.mean(filt)
 
    if np.all(raw == 0) or np.all(filt == 0) or np.isnan(raw).any() or np.isnan(filt).any():
        return 0.0

    corr = np.correlate(filt, raw, mode="full")
    lags = np.arange(-len(raw) + 1, len(raw))
    best_lag_idx = lags[np.argmax(corr)]
 
    dt = np.mean(np.diff(t))
    return best_lag_idx * dt
 
def analyze(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ANALYZE ERROR] Cannot find file: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path).dropna()
        if len(df) < 10:
            print("[ANALYZE WARNING] < 10 frames - Skipping analysis.")
            return
            
        t = df["frame_timestamp_s"].to_numpy()
        
        report_lines = []
        report_lines.append(f"=== BÁO CÁO PHÂN TÍCH BỘ LỌC ===")
        report_lines.append(f"Tệp nguồn: {os.path.basename(csv_path)}")
        report_lines.append(f"Số lượng Frame hợp lệ: {len(df)}")
        report_lines.append(f"----------------------------------\n")
    
        report_lines.append("-- Độ trễ (Lag) theo từng trục vị trí --")
        for axis in ["x", "y", "z"]:
            lag = estimate_lag(t, df[f"raw_{axis}"], df[f"filt_{axis}"])
            report_lines.append(f"  Trục {axis.upper()}: {lag*1000:.1f} ms")
    
        report_lines.append("\n-- Độ trễ (Lag) theo từng thành phần Quaternion --")
        for c in ["qw", "qx", "qy", "qz"]:
            lag = estimate_lag(t, df[f"raw_{c}"], df[f"filt_{c}"])
            report_lines.append(f"  {c}: {lag*1000:.1f} ms")
    
        report_lines.append("\n-- Độ rung (Jitter/Std) - Raw vs Filtered --")
        for axis in ["x", "y", "z"]:
            raw_std = df[f"raw_{axis}"].std() * 1000  # mm
            filt_std = df[f"filt_{axis}"].std() * 1000
            report_lines.append(f"  Trục {axis.upper()}: raw_std = {raw_std:.2f} mm  |  filt_std = {filt_std:.2f} mm")
        
        full_report = "\n".join(report_lines)
        
        print(f"\n{full_report}\n")
        
        txt_path = csv_path.replace(".csv", "_analysis.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_report)
            
        print(f"[ANALYZE INFO] Saved at: {txt_path}")
        
    except Exception as e:
        print(f"[ANALYZE ERROR] Error while analyzing: {e}")
 
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/RT_20260708_120000.csv"
    analyze(path)