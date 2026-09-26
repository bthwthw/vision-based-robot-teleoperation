import os
import sys
import numpy as np
import pandas as pd

def analyze_vision_frequency(csv_path):
    if not os.path.exists(csv_path):
        alt_path = csv_path.replace(".csv", "_vision.csv")
        if os.path.exists(alt_path):
            print(f"[FREQ ANALYZER INFO] {csv_path} not found, using {alt_path} instead")
            csv_path = alt_path
        else:
            print(f"[FREQ ANALYZER ERROR] File not found: {csv_path}")
            return

    df = pd.read_csv(csv_path)
    if len(df) < 5:
        print("[FREQ ANALYZER WARN] Empty vision CSV")
        return

    report_lines = []

    def log_print(msg=""):
        print(msg)
        report_lines.append(msg)

    log_print("\n=== VISION (PERCEPTION) LOOP FREQUENCY ===")
    log_print(f"File   : {os.path.basename(csv_path)}")
    log_print(f"Total Rows : {len(df)}")

    df['dt_s'] = df['wall_time_s'].diff()
    df_valid = df.dropna(subset=['dt_s']).copy()
    df_valid = df_valid[df_valid['dt_s'] > 0]
    df_valid['dt_ms'] = df_valid['dt_s'] * 1000.0
    df_valid['freq_hz'] = 1.0 / df_valid['dt_s']

    log_print(f"  Mean Frequency  : {df_valid['freq_hz'].mean():.2f} Hz")
    log_print(f"  Median Frequency: {df_valid['freq_hz'].median():.2f} Hz")
    log_print(f"  Min Frequency   : {df_valid['freq_hz'].min():.2f} Hz")
    log_print(f"  Max Frequency   : {df_valid['freq_hz'].max():.2f} Hz")
    log_print("  --")
    log_print(f"  Mean Loop Time  : {df_valid['dt_ms'].mean():.2f} ms")
    log_print(f"  Median Loop Time: {df_valid['dt_ms'].median():.2f} ms")
    log_print(f"  Max Loop Time   : {df_valid['dt_ms'].max():.2f} ms (Spike)")
    log_print(f"  Std Loop Time   : {df_valid['dt_ms'].std():.2f} ms")

    spike_count = np.sum(df_valid['dt_ms'] > 100.0)
    log_print(f"  Lag Spikes (>100ms) : {spike_count} frames ({(spike_count/len(df_valid))*100:.1f}%)")

    # Liệt kê các mốc thời gian bị spike, dùng để đối chiếu chéo với record.csv (kiểm tra GIL contention)
    spikes = df_valid[df_valid['dt_ms'] > 100.0][['wall_time_s', 'dt_ms']]
    if not spikes.empty:
        log_print("\n[VISION SPIKE TIMESTAMPS] (wall_time_s, dt_ms) - dùng để so khớp với record.csv")
        for _, row in spikes.head(30).iterrows():
            log_print(f"  t={row['wall_time_s']:.3f}s  dt={row['dt_ms']:.1f}ms")
        if len(spikes) > 30:
            log_print(f"  ... và {len(spikes) - 30} spike khác")

    full_report = "\n".join(report_lines)
    txt_path = csv_path.replace(".csv", "_vision_freq_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n[FREQ ANALYZER INFO] Saved at: {txt_path}\n")

    return df_valid  # trả về để dùng cho correlate_spikes bên dưới


def correlate_spikes(vision_csv, record_csv, window_s=0.15):
    """
    Kiểm tra giả thuyết GIL contention: với mỗi tick record (PyBullet) chậm,
    xem có spike vision (perception) xảy ra gần đúng thời điểm đó không.
    Nếu trùng khớp nhiều -> xác nhận PyBullet capture đang block cả GIL,
    không chỉ ảnh hưởng riêng thread của nó.
    """
    if not (os.path.exists(vision_csv) and os.path.exists(record_csv)):
        print("[CORRELATE ERROR] Thiếu file vision hoặc record CSV")
        return

    df_vision = pd.read_csv(vision_csv)
    df_record = pd.read_csv(record_csv)

    df_vision['dt_ms'] = df_vision['wall_time_s'].diff() * 1000.0
    df_record['dt_ms'] = df_record['wall_time_s'].diff() * 1000.0

    vision_spikes = df_vision[df_vision['dt_ms'] > 100.0]
    record_spikes = df_record[df_record['dt_ms'] > 100.0]  # hoặc dùng t_tick_total_ms nếu có

    if vision_spikes.empty or record_spikes.empty:
        print("[CORRELATE INFO] Không đủ spike ở 1 trong 2 file để đối chiếu")
        return

    matched = 0
    print("\n=== ĐỐI CHIẾU SPIKE: vision.csv vs record.csv (GIL contention check) ===")
    for _, rrow in record_spikes.iterrows():
        t_rec = rrow['wall_time_s']
        close = vision_spikes[(vision_spikes['wall_time_s'] - t_rec).abs() <= window_s]
        if not close.empty:
            matched += 1
            print(f"  record spike t={t_rec:.3f}s (dt={rrow['dt_ms']:.0f}ms) "
                  f"<-> vision spike t={close.iloc[0]['wall_time_s']:.3f}s "
                  f"(dt={close.iloc[0]['dt_ms']:.0f}ms)")

    pct = 100 * matched / len(record_spikes)
    print(f"\n  {matched}/{len(record_spikes)} record-spike trùng khớp vision-spike trong ±{window_s*1000:.0f}ms "
          f"({pct:.1f}%)")
    if pct > 50:
        print("  -> Khả năng cao: PyBullet capture đang block GIL, ảnh hưởng cả thread Perception.")
    else:
        print("  -> Không thấy tương quan rõ -> 2 vấn đề có thể độc lập nhau.")


def correlate_spikes(vision_csv, record_csv, window_s=0.15):
    """
    Kiểm tra giả thuyết GIL contention: với mỗi tick record (PyBullet) chậm,
    xem có spike vision (perception) xảy ra gần đúng thời điểm đó không.
    Nếu trùng khớp nhiều -> xác nhận PyBullet capture đang block cả GIL,
    không chỉ ảnh hưởng riêng thread của nó.
    """
    if not (os.path.exists(vision_csv) and os.path.exists(record_csv)):
        print("[CORRELATE ERROR] Thiếu file vision hoặc record CSV")
        return

    df_vision = pd.read_csv(vision_csv)
    df_record = pd.read_csv(record_csv)

    df_vision['dt_ms'] = df_vision['wall_time_s'].diff() * 1000.0
    df_record['dt_ms'] = df_record['wall_time_s'].diff() * 1000.0

    vision_spikes = df_vision[df_vision['dt_ms'] > 100.0]
    record_spikes = df_record[df_record['dt_ms'] > 100.0]  # hoặc dùng t_tick_total_ms nếu có

    if vision_spikes.empty or record_spikes.empty:
        print("[CORRELATE INFO] Không đủ spike ở 1 trong 2 file để đối chiếu")
        return

    matched = 0
    print("\n=== ĐỐI CHIẾU SPIKE: vision.csv vs record.csv (GIL contention check) ===")
    for _, rrow in record_spikes.iterrows():
        t_rec = rrow['wall_time_s']
        close = vision_spikes[(vision_spikes['wall_time_s'] - t_rec).abs() <= window_s]
        if not close.empty:
            matched += 1
            print(f"  record spike t={t_rec:.3f}s (dt={rrow['dt_ms']:.0f}ms) "
                  f"<-> vision spike t={close.iloc[0]['wall_time_s']:.3f}s "
                  f"(dt={close.iloc[0]['dt_ms']:.0f}ms)")

    pct = 100 * matched / len(record_spikes)
    print(f"\n  {matched}/{len(record_spikes)} record-spike trùng khớp vision-spike trong ±{window_s*1000:.0f}ms "
          f"({pct:.1f}%)")
    if pct > 50:
        print("  -> Khả năng cao: PyBullet capture đang block GIL, ảnh hưởng cả thread Perception.")
    else:
        print("  -> Không thấy tương quan rõ -> 2 vấn đề có thể độc lập nhau.")

def analyze_record_frequency(csv_path, target_hz=15.0):
    if not os.path.exists(csv_path):
        print(f"[FREQ ANALYZER ERROR] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if len(df) < 5:
        print("[FREQ ANALYZER WARN] Empty record CSV")
        return

    report_lines = []

    def log_print(msg=""):
        print(msg)
        report_lines.append(msg)

    df['dt_s'] = df['wall_time_s'].diff()
    df_valid = df.dropna(subset=['dt_s'])
    df_valid = df_valid[df_valid['dt_s'] > 0].copy()
    df_valid['freq_hz'] = 1.0 / df_valid['dt_s']

    target_period_ms = (1.0 / target_hz) * 1000.0
    df_valid['dt_ms'] = df_valid['dt_s'] * 1000.0
    df_valid['deviation_ms'] = (df_valid['dt_ms'] - target_period_ms).abs()

    log_print("\n=== RECORD TICK FREQUENCY (dùng cho LeRobot dataset) ===")
    log_print(f"File   : {os.path.basename(csv_path)}")
    log_print(f"Total Rows : {len(df)}")
    log_print(f"Target: {target_hz} Hz ({target_period_ms:.2f} ms/tick)")
    log_print(f"  Mean Hz    : {df_valid['freq_hz'].mean():.2f}")
    log_print(f"  Std Hz     : {df_valid['freq_hz'].std():.2f}")
    log_print(f"  Min Hz     : {df_valid['freq_hz'].min():.2f}")
    log_print(f"  Mean jitter (|dt - target|): {df_valid['deviation_ms'].mean():.2f} ms")
    log_print(f"  Max jitter  : {df_valid['deviation_ms'].max():.2f} ms")

    within_tol = np.sum(df_valid['deviation_ms'] <= 3.0)
    log_print(f"  Ticks within ±3ms of target: {within_tol}/{len(df_valid)} "
              f"({100*within_tol/len(df_valid):.1f}%)")

    cap_cols = ["t_wrist_capture_ms", "t_world_capture_ms", "t_tick_total_ms"]
    if all(c in df.columns for c in cap_cols):
        log_print("\n[CAPTURE COST PER TICK]")
        for c in cap_cols:
            vals = df[c].dropna().to_numpy()
            if len(vals):
                log_print(f"  {c:22s}: mean={np.mean(vals):.2f}ms max={np.max(vals):.2f}ms")

    full_report = "\n".join(report_lines)
    txt_path = csv_path.replace(".csv", "_record_freq_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n[FREQ ANALYZER INFO] Saved at: {txt_path}\n")

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

            median_latency = np.median(latency)
            adaptive_threshold = max(100.0, median_latency * 1.5) 
            curobo_spike_count = np.sum(latency > adaptive_threshold)
            log_print(f"  IK Spikes (>{adaptive_threshold:.0f}ms, adaptive) : {curobo_spike_count} frames "
                    f"({(curobo_spike_count/len(latency))*100:.1f}%)")

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
    record_path = sys.argv[1] if len(sys.argv) > 1 else "logs/PB_20260904_132118_record.csv"
    vision_path = record_path.replace("_record.csv", "_vision.csv")

    # analyze_record_frequency(record_path)
    # analyze_vision_frequency(vision_path)
    # correlate_spikes(vision_path, record_path)

    import pandas as pd
    df = pd.read_csv("logs/PB_20260904_132118_record.csv")

    action_cols = ["action_1","action_2","action_3","action_4","action_5","action_6"]
    same_as_prev = (df[action_cols].diff().abs().sum(axis=1) < 1e-6)
    duplicate_rate = same_as_prev.mean() * 100
    print(f"Tỷ lệ record tick có action trùng hệt tick trước: {duplicate_rate:.1f}%")