import os
import sys
import numpy as np
import pandas as pd

def analyze_frequency(csv_path):
    if not os.path.exists(csv_path):
        print(f"[FREQ ANALYZER ERROR] File not found: {csv_path}")
        return

    try:
        # Load file (Không dropna ngay để giữ nguyên tính liên tục của các frame)
        df = pd.read_csv(csv_path)
        if len(df) < 5:
            print("[FREQ ANALYZER WARN] Empty CSV")
            return

        report_lines = []
        def log_print(msg=""):
            print(msg)
            report_lines.append(msg)

        log_print(f"\n=== SYSTEM FREQUENCY & PERFORMANCE ANALYSIS ===")
        log_print(f"File   : {os.path.basename(csv_path)}")
        log_print(f"Total Rows : {len(df)}")
        
        # 1. Tổng quan chu kỳ hệ thống (System Loop dt & Frequency)
        df['dt_s'] = df['wall_time_s'].diff()
        df_valid_dt = df.dropna(subset=['dt_s']).copy()
        
        # Loại bỏ các frame có dt <= 0 để tránh lỗi chia cho 0
        df_valid_dt = df_valid_dt[df_valid_dt['dt_s'] > 0]
        df_valid_dt['dt_ms'] = df_valid_dt['dt_s'] * 1000.0
        df_valid_dt['freq_hz'] = 1.0 / df_valid_dt['dt_s']

        log_print("\n[OVERALL SYSTEM LOOP FREQUENCY]")
        log_print(f"  Mean Frequency  : {df_valid_dt['freq_hz'].mean():.2f} Hz")
        log_print(f"  Median Frequency: {df_valid_dt['freq_hz'].median():.2f} Hz")
        log_print(f"  Min Frequency   : {df_valid_dt['freq_hz'].min():.2f} Hz")
        log_print(f"  Max Frequency   : {df_valid_dt['freq_hz'].max():.2f} Hz")
        log_print(f"  --")
        log_print(f"  Mean Loop Time  : {df_valid_dt['dt_ms'].mean():.2f} ms")
        log_print(f"  Median Loop Time: {df_valid_dt['dt_ms'].median():.2f} ms")
        log_print(f"  Max Loop Time   : {df_valid_dt['dt_ms'].max():.2f} ms (Spike)")
        log_print(f"  Std Loop Time   : {df_valid_dt['dt_ms'].std():.2f} ms")

        # Đếm số lần sụt giảm khung hình (Lag spikes) - ví dụ vòng lặp mất quá 100ms (< 10Hz)
        spike_count = np.sum(df_valid_dt['dt_ms'] > 100.0)
        log_print(f"  Lag Spikes (>100ms) : {spike_count} frames ({(spike_count/len(df_valid_dt))*100:.1f}%)")

        # 2. Hiệu năng tính toán cuRobo IK
        df_curobo = df.dropna(subset=["curobo_time_ms"]).copy()
        if not df_curobo.empty:
            latency = df_curobo["curobo_time_ms"].to_numpy()
            log_print("\n[CUROBO IK EXECUTION TIME]")
            log_print(f"  Mean Latency    : {np.mean(latency):.2f} ms")
            log_print(f"  Median Latency  : {np.median(latency):.2f} ms")
            log_print(f"  Max Latency     : {np.max(latency):.2f} ms")
            log_print(f"  Min Latency     : {np.min(latency):.2f} ms")
            log_print(f"  Std Latency     : {np.std(latency):.2f} ms")
            
            curobo_spike_count = np.sum(latency > 100.0)
            log_print(f"  IK Spikes (>100ms)  : {curobo_spike_count} frames ({(curobo_spike_count/len(latency))*100:.1f}%)")
            
            # 3. Phân bổ thời gian (Time Distribution) - Curobo vs Outside Time
            # Outside Time = Tổng thời gian vòng lặp - Thời gian chạy CuRobo
            df_curobo['dt_ms'] = df_curobo['wall_time_s'].diff() * 1000.0
            df_curobo_valid = df_curobo.dropna(subset=['dt_ms']).copy()
            df_curobo_valid['outside_time_ms'] = df_curobo_valid['dt_ms'] - df_curobo_valid['curobo_time_ms']
            
            # Cắt các giá trị âm (trường hợp wall_time lệch nhịp nhỏ)
            df_curobo_valid['outside_time_ms'] = df_curobo_valid['outside_time_ms'].clip(lower=0)
            
            out_time = df_curobo_valid['outside_time_ms'].to_numpy()
            log_print("\n[OUTSIDE CUROBO TIME (Shared Mem, Sleep, Logging)]")
            log_print(f"  Mean Outside Time : {np.mean(out_time):.2f} ms")
            log_print(f"  Median Outside Time:{np.median(out_time):.2f} ms")
            log_print(f"  Max Outside Time  : {np.max(out_time):.2f} ms")
            log_print(f"  Std Outside Time  : {np.std(out_time):.2f} ms")
            
            out_spike_count = np.sum(out_time > 50.0)
            log_print(f"  Outside Spikes (>50ms): {out_spike_count} frames ({(out_spike_count/len(out_time))*100:.1f}%)")

        full_report = "\n".join(report_lines)
        txt_path = csv_path.replace(".csv", "_freq_analysis.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n[FREQ ANALYZER INFO] Saved at: {txt_path}\n")

    except Exception as e:
        print(f"[FREQ ANALYZER ERROR] Analysis failed: {e}\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/RT_20260807_142820.csv"
    analyze_frequency(path)