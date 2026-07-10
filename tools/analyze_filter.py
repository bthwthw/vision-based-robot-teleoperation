import sys
import os
import numpy as np
import pandas as pd

def _fix_quaternion_continuity(q):
    """
    Quaternions q and -q represent the SAME rotation (double cover).
    This function forces consecutive quaternions to the same sign branch to ensure continuity.
    """
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q

def estimate_lag(t, raw, filt, max_lag_s=0.5):
    """
    Estimate the lag (seconds) between raw and filtered signals using cross-correlation.
    Search is bounded by +-max_lag_s to prevent false peak detection.
    Returns a positive value if the filtered signal lags behind the raw signal.
    """
    raw = np.asarray(raw, dtype=float) - np.mean(raw)
    filt = np.asarray(filt, dtype=float) - np.mean(filt)
    
    if np.all(raw == 0) or np.all(filt == 0) or np.isnan(raw).any() or np.isnan(filt).any():
        return 0.0

    dt = np.mean(np.diff(t))
    if dt <= 0:
        return 0.0
        
    max_lag_n = max(1, int(max_lag_s / dt))

    corr = np.correlate(filt, raw, mode="full")
    lags = np.arange(-len(raw) + 1, len(raw))
    
    mask = np.abs(lags) <= max_lag_n
    if not np.any(mask):
        return 0.0
        
    best_lag_idx = lags[mask][np.argmax(corr[mask])]
    return best_lag_idx * dt

def analyze(csv_path):
    """
    Analyze the CSV data, print to console, and export a sidecar text report.
    """
    if not os.path.exists(csv_path):
        print(f"[ANALYZER ERROR] File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path).dropna()
        if len(df) < 10:
            print("[ANALYZER WARN] < 10 frames. Skipping analysis.")
            return

        t = df["frame_timestamp_s"].to_numpy()
        dt_arr = np.diff(t)
        
        report_lines = []
        def log_print(msg=""):
            print(msg)
            report_lines.append(msg)

        # Header
        log_print(f"\n=== FILTER ANALYSIS ===")
        log_print(f"File   : {os.path.basename(csv_path)}")
        log_print(f"Frames : {len(df)}")
        log_print(f"FPS    : mean={np.mean(dt_arr)*1000:.1f}ms | std={np.std(dt_arr)*1000:.1f}ms | max={np.max(dt_arr)*1000:.1f}ms")
        
        if np.max(dt_arr) > 3 * np.mean(dt_arr):
            log_print("[ANALYZER WARN] High frame time variance detected. Metrics may be degraded.")
            
        # Position Lag
        log_print("\n[POSITION LAG]")
        for axis in ["x", "y", "z"]:
            lag = estimate_lag(t, df[f"raw_{axis}"], df[f"filt_{axis}"])
            log_print(f"  {axis.upper()}: {lag*1000:.1f} ms")

        # Orientation Lag
        log_print("\n[ORIENTATION LAG]")
        raw_q = _fix_quaternion_continuity(df[["raw_qw", "raw_qx", "raw_qy", "raw_qz"]].to_numpy())
        filt_q = _fix_quaternion_continuity(df[["filt_qw", "filt_qx", "filt_qy", "filt_qz"]].to_numpy())
        
        ref = raw_q[0]
        ang_raw = 2 * np.arccos(np.clip(np.abs(raw_q @ ref), -1.0, 1.0))
        ang_filt = 2 * np.arccos(np.clip(np.abs(filt_q @ ref), -1.0, 1.0))
        
        lag_ori = estimate_lag(t, ang_raw, ang_filt)
        log_print(f"  Lag: {lag_ori*1000:.1f} ms")

        # Jitter Reduction
        log_print("\n[JITTER REDUCTION]")
        for axis in ["x", "y", "z"]:
            jr = np.std(np.diff(df[f"raw_{axis}"])) * 1000
            jf = np.std(np.diff(df[f"filt_{axis}"])) * 1000
            reduction = 100 * (1 - jf / jr) if jr > 0 else 0
            log_print(f"  {axis.upper()}  : raw={jr:.2f}mm | filt={jf:.2f}mm (-{reduction:.0f}%)")

        dot_raw = np.clip(np.sum(raw_q[:-1] * raw_q[1:], axis=1), -1.0, 1.0)
        dot_filt = np.clip(np.sum(filt_q[:-1] * filt_q[1:], axis=1), -1.0, 1.0)
        step_raw = np.degrees(2 * np.arccos(np.abs(dot_raw)))
        step_filt = np.degrees(2 * np.arccos(np.abs(dot_filt)))
        
        mean_step_raw = np.mean(step_raw)
        reduction_ori = 100 * (1 - np.mean(step_filt) / mean_step_raw) if mean_step_raw > 0 else 0
        log_print(f"  Ori: raw={mean_step_raw:.2f}deg | filt={np.mean(step_filt):.2f}deg (-{reduction_ori:.0f}%)")

        # Outlier Check
        omega_raw_deg_s = step_raw / dt_arr
        log_print(f"\n[KINEMATIC CHECK]")
        log_print(f"  Max AngVel: {np.max(omega_raw_deg_s):.0f} deg/s (Mean: {np.mean(omega_raw_deg_s):.1f} deg/s)")
        if np.max(omega_raw_deg_s) > 1000:
            log_print(" [ANALYZER WARN] AngVel > 1000 deg/s. Possible outlier/misdetection.")

        # Export to TXT
        full_report = "\n".join(report_lines)
        txt_path = csv_path.replace(".csv", "_analysis.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_report)
            
        print(f"\n[ANALYZER INFO] Saved analysis to: {txt_path}\n")
        
    except Exception as e:
        print(f"[ANALYZER ERROR] Analysis failed: {e}\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/session1.csv"
    analyze(path)