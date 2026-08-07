import os
import sys
import numpy as np
import pandas as pd


def analyze_frequency(csv_path):
    if not os.path.exists(csv_path):
        alt_path = csv_path.replace(".csv", "_control.csv")
        if os.path.exists(alt_path):
            print(f"[FREQ ANALYZER INFO] {csv_path} not found, using {alt_path} instead")
            csv_path = alt_path
        else:
            print(f"[FREQ ANALYZER ERROR] File not found: {csv_path}")
            return

    try:
        df = pd.read_csv(csv_path)
        if len(df) < 5:
            print("[FREQ ANALYZER WARN] Empty CSV")
            return

        report_lines = []

        def log_print(msg=""):
            print(msg)
            report_lines.append(msg)

        log_print("\n=== SYSTEM FREQUENCY & PERFORMANCE ANALYSIS ===")
        log_print(f"File   : {os.path.basename(csv_path)}")
        log_print(f"Total Rows : {len(df)}")

        df['dt_s'] = df['wall_time_s'].diff()
        df_valid_dt = df.dropna(subset=['dt_s']).copy()
        df_valid_dt = df_valid_dt[df_valid_dt['dt_s'] > 0]
        df_valid_dt['dt_ms'] = df_valid_dt['dt_s'] * 1000.0
        df_valid_dt['freq_hz'] = 1.0 / df_valid_dt['dt_s']

        log_print("\n[CONTROL LOOP FREQUENCY]")
        log_print(f"  Mean Frequency  : {df_valid_dt['freq_hz'].mean():.2f} Hz")
        log_print(f"  Median Frequency: {df_valid_dt['freq_hz'].median():.2f} Hz")
        log_print(f"  Min Frequency   : {df_valid_dt['freq_hz'].min():.2f} Hz")
        log_print(f"  Max Frequency   : {df_valid_dt['freq_hz'].max():.2f} Hz")
        log_print("  --")
        log_print(f"  Mean Loop Time  : {df_valid_dt['dt_ms'].mean():.2f} ms")
        log_print(f"  Median Loop Time: {df_valid_dt['dt_ms'].median():.2f} ms")
        log_print(f"  Max Loop Time   : {df_valid_dt['dt_ms'].max():.2f} ms (Spike)")
        log_print(f"  Std Loop Time   : {df_valid_dt['dt_ms'].std():.2f} ms")

        spike_count = np.sum(df_valid_dt['dt_ms'] > 100.0)
        log_print(f"  Lag Spikes (>100ms) : {spike_count} frames ({(spike_count/len(df_valid_dt))*100:.1f}%)")

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

        timer_cols = ["t_read_pose_ms", "t_pb_query_ms", "t_sleep_ms", "t_loop_total_ms"]
        if all(c in df.columns for c in timer_cols):
            df_t = df.dropna(subset=timer_cols).copy()
            if not df_t.empty:
                log_print("\n[STEP-BY-STEP TIMER BREAKDOWN]")
                for col, label in [
                    ("t_read_pose_ms", "Read shared_pose"),
                    ("t_pb_query_ms", "Read shared_pb_state (PyBullet)"),
                    ("t_sleep_ms", "time.sleep(0.01)"),
                    ("t_loop_total_ms", "TOTAL loop (perf_counter)"),
                ]:
                    vals = df_t[col].to_numpy()
                    log_print(f"  {label:32s}: mean={np.mean(vals):7.2f}ms  "
                              f"median={np.median(vals):7.2f}ms  max={np.max(vals):8.2f}ms")

                if "curobo_time_ms" in df_t.columns:
                    accounted = (df_t["t_read_pose_ms"] + df_t["curobo_time_ms"].fillna(0)
                                 + df_t["t_pb_query_ms"] + df_t["t_sleep_ms"])
                    unaccounted = (df_t["t_loop_total_ms"] - accounted).clip(lower=0)
                    log_print(f"  {'Unaccounted':32s}: mean={unaccounted.mean():7.2f}ms  "
                              f"median={unaccounted.median():7.2f}ms  max={unaccounted.max():8.2f}ms")
        else:
            log_print("\n[FREQ ANALYZER ERROR] Timer columns missing in CSV, skipping timer breakdown")

        full_report = "\n".join(report_lines)
        txt_path = csv_path.replace(".csv", "_freq_analysis.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n[FREQ ANALYZER INFO] Saved at: {txt_path}\n")

    except Exception as e:
        print(f"[FREQ ANALYZER ERROR] Analysis failed: {e}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/RT_20260807_163625_control.csv"
    analyze_frequency(path)