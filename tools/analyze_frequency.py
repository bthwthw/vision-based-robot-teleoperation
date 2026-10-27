import os
import re
import sys
import traceback
import numpy as np
import pandas as pd

TARGET_HZ = 10.0
TOLERANCE_MS = 3.0
LONG_FRAME_FACTOR = 3.0
CORR_WINDOW_S = 0.1
ACTION_COLS = [f"action_{i}" for i in range(1, 7)]
ACTION_TOL = 1e-6
MAX_LISTED = 20


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        print(msg)
        self.lines.append(msg)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\n[INFO] Saved: {path}\n")


def _intervals(df, col="wall_time_s"):
    t = pd.to_numeric(df[col], errors="coerce").to_numpy()
    d = pd.DataFrame({"idx": np.arange(len(t)), "t_prev": np.r_[np.nan, t[:-1]], "t": t})
    d["dt_s"] = d["t"] - d["t_prev"]
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    bad = int((d["dt_s"] <= 0).sum())
    d = d[d["dt_s"] > 0].copy()
    d["dt_ms"] = d["dt_s"] * 1000.0
    return d.reset_index(drop=True), bad


def _long_threshold(dt_ms):
    return LONG_FRAME_FACTOR * float(np.median(dt_ms))


def _loop_report(log, d, bad):
    dt = d["dt_ms"].to_numpy()
    thr = _long_threshold(dt)
    short = d[d["dt_ms"] <= thr]
    n_long = len(d) - len(short)

    log(f"  Intervals: {len(d)} | dropped non-positive: {bad}")
    log(f"  Rate: overall={len(d) / d['dt_s'].sum():.2f} | excl. long={len(short) / short['dt_s'].sum():.2f} | "
        f"median={1000 / np.median(dt):.2f} | p5={1000 / np.percentile(dt, 95):.2f} Hz")
    log(f"  dt  : mean={dt.mean():.1f} | med={np.median(dt):.1f} | p95={np.percentile(dt, 95):.1f} | "
        f"p99={np.percentile(dt, 99):.1f} | max={dt.max():.1f} | std={dt.std():.1f} ms")
    log(f"  Long frames (> {LONG_FRAME_FACTOR:.0f}x median = {thr:.0f}ms): {n_long} ({100 * n_long / len(dt):.1f}%) | "
        f"time in long frames: {d.loc[d['dt_ms'] > thr, 'dt_s'].sum():.1f}s of {d['dt_s'].sum():.1f}s")
    return thr


def _list_long(log, d, thr, df=None, timed_col=None):
    long_d = d[d["dt_ms"] > thr]
    if long_d.empty:
        return
    log("\n[LONG FRAMES] (t_start -> t_end, dt)")
    for _, r in long_d.head(MAX_LISTED).iterrows():
        line = f"  {r['t_prev']:.3f} -> {r['t']:.3f}s  dt={r['dt_ms']:.0f}ms"
        if df is not None and timed_col in df.columns:
            timed = float(df.loc[int(r["idx"]), timed_col])
            where = "inside timed section" if timed > 0.5 * r["dt_ms"] else "OUTSIDE timed section"
            line += f" | {timed_col}={timed:.0f}ms ({where})"
        log(line)
    if len(long_d) > MAX_LISTED:
        log(f"  ... +{len(long_d) - MAX_LISTED} more")


def analyze_vision_frequency(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        print("[WARN] Vision CSV too short")
        return

    log = Report()
    log("\n=== VISION LOOP FREQUENCY ===")
    log(f"File: {os.path.basename(csv_path)} | rows: {len(df)}")

    d, bad = _intervals(df)
    thr = _loop_report(log, d, bad)
    _list_long(log, d, thr)
    log.save(csv_path.replace(".csv", "_vision_freq_analysis.txt"))
    return d


def correlate_long_frames(vision_csv, record_csv, window_s=CORR_WINDOW_S):
    v, _ = _intervals(pd.read_csv(vision_csv))
    r, _ = _intervals(pd.read_csv(record_csv))
    vl = v[v["dt_ms"] > _long_threshold(v["dt_ms"])].reset_index(drop=True)
    rl = r[r["dt_ms"] > _long_threshold(r["dt_ms"])].reset_index(drop=True)

    if vl.empty or rl.empty:
        print("[INFO] No long frames in vision or record; nothing to correlate")
        return

    def overlaps(a, b):
        a0, a1 = a["t_prev"].to_numpy()[:, None], a["t"].to_numpy()[:, None]
        b0, b1 = b["t_prev"].to_numpy()[None, :], b["t"].to_numpy()[None, :]
        return ((a0 - window_s) <= b1) & ((b0 - window_s) <= a1)

    rv = overlaps(rl, vl)
    rec_hit = rv.any(axis=1)
    vis_hit = rv.any(axis=0)

    span = min(v["t"].max(), r["t"].max()) - max(v["t_prev"].min(), r["t_prev"].min())
    if span <= 0:
        print("[WARN] Vision and record time ranges do not overlap")
        return
    chance_r = min(1.0, (vl["dt_s"].sum() + len(vl) * (rl["dt_s"].mean() + 2 * window_s)) / span)
    chance_v = min(1.0, (rl["dt_s"].sum() + len(rl) * (vl["dt_s"].mean() + 2 * window_s)) / span)

    log = Report()
    log("\n=== LONG-FRAME OVERLAP: vision vs record ===")
    log(f"Window: +/-{window_s * 1000:.0f}ms | long frames: vision={len(vl)}, record={len(rl)}")
    log(f"  record -> vision: {int(rec_hit.sum())}/{len(rl)} ({100 * rec_hit.mean():.1f}%) | chance~{100 * chance_r:.1f}%")
    log(f"  vision -> record: {int(vis_hit.sum())}/{len(vl)} ({100 * vis_hit.mean():.1f}%) | chance~{100 * chance_v:.1f}%")

    log("\n[EVENTS] record long frame <-> vision long frame")
    for i, row in rl.head(MAX_LISTED).iterrows():
        js = np.where(rv[i])[0]
        rec = f"rec {row['t_prev']:.3f}->{row['t']:.3f}s ({row['dt_ms']:.0f}ms)"
        if len(js) == 0:
            log(f"  {rec} <-> none")
        else:
            vis = "; ".join(f"vis {vl.loc[j, 't_prev']:.3f}->{vl.loc[j, 't']:.3f}s ({vl.loc[j, 'dt_ms']:.0f}ms)" for j in js)
            log(f"  {rec} <-> {vis}")

    log("  Overlap above chance suggests a shared stall (correlation only, cause not proven)")
    log.save(record_csv.replace(".csv", "_overlap_analysis.txt"))


def _log_action_duplicates(log, df):
    cols = [c for c in ACTION_COLS if c in df.columns]
    if not cols:
        return
    a = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(a) < 2:
        return
    same = a.diff().abs().iloc[1:].max(axis=1) < ACTION_TOL
    log("\n[ACTION REPEATS]")
    log(f"  Ticks identical to previous: {int(same.sum())}/{len(same)} ({100 * same.mean():.1f}%)")


def analyze_record_frequency(csv_path, target_hz=TARGET_HZ):
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        print("[WARN] Record CSV too short")
        return

    log = Report()
    target_ms = 1000.0 / target_hz
    log("\n=== RECORD TICK FREQUENCY ===")
    log(f"File: {os.path.basename(csv_path)} | rows: {len(df)}")
    log(f"Target: {target_hz} Hz ({target_ms:.1f} ms)")

    d, bad = _intervals(df)
    thr = _loop_report(log, d, bad)

    dt = d["dt_ms"].to_numpy()
    dev = np.abs(dt - target_ms)
    within = int((dev <= TOLERANCE_MS).sum())
    log(f"  Median dt vs target: {np.median(dt) - target_ms:+.1f} ms")
    log(f"  Within +/-{TOLERANCE_MS:.0f}ms of target: {within}/{len(d)} ({100 * within / len(d):.1f}%)")

    if "t_tick_total_ms" in df.columns:
        tick = pd.to_numeric(df["t_tick_total_ms"], errors="coerce").to_numpy()[d["idx"].to_numpy()]
        untimed = dt - tick
        untimed = untimed[np.isfinite(untimed)]
        short = untimed[dt[: len(untimed)] <= thr] if len(untimed) == len(dt) else untimed
        log(f"  Untimed per tick (dt - t_tick_total), excl. long: med={np.median(short):.1f} | "
            f"p95={np.percentile(short, 95):.1f} ms")

    cap_cols = ["t_wrist_capture_ms", "t_world_capture_ms", "t_tick_total_ms"]
    if all(c in df.columns for c in cap_cols):
        log("\n[CAPTURE COST]")
        for c in cap_cols:
            vals = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
            if len(vals):
                log(f"  {c:20s}: mean={vals.mean():.2f} | p95={np.percentile(vals, 95):.2f} | max={vals.max():.2f} ms")

    _list_long(log, d, thr, df, "t_tick_total_ms")
    _log_action_duplicates(log, df)
    log.save(csv_path.replace(".csv", "_record_freq_analysis.txt"))


def analyze_frequency(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        print("[WARN] Control CSV too short")
        return

    log = Report()
    log("\n=== CONTROL LOOP FREQUENCY ===")
    log(f"File: {os.path.basename(csv_path)} | rows: {len(df)}")

    d, bad = _intervals(df)
    thr = _loop_report(log, d, bad)
    n_long = int((d["dt_ms"] > thr).sum())
    log(f"  Active segments (split at long frames): {n_long + 1}; use 'excl. long' rate for loop speed")
    _list_long(log, d, thr)

    if "curobo_time_ms" in df.columns:
        lat = pd.to_numeric(df["curobo_time_ms"], errors="coerce").dropna().to_numpy()
        if len(lat):
            log("\n[CUROBO IK TIME]")
            log(f"  mean={lat.mean():.2f} | med={np.median(lat):.2f} | p95={np.percentile(lat, 95):.2f} | "
                f"max={lat.max():.2f} | min={lat.min():.2f} | std={lat.std():.2f} ms")
            lthr = _long_threshold(lat)
            n = int((lat > lthr).sum())
            log(f"  Slow solves (> {LONG_FRAME_FACTOR:.0f}x median = {lthr:.1f}ms): {n} ({100 * n / len(lat):.1f}%)")

    timers = [("t_read_pose_ms", "read_pose"), ("t_pb_query_ms", "pb_query"),
              ("t_sleep_ms", "sleep"), ("t_loop_total_ms", "loop_total")]
    if all(c in df.columns for c, _ in timers):
        t = df[[c for c, _ in timers]].apply(pd.to_numeric, errors="coerce").dropna()
        if not t.empty:
            log("\n[TIMER BREAKDOWN]")
            for c, label in timers:
                v = t[c].to_numpy()
                log(f"  {label:11s}: mean={v.mean():7.2f} | med={np.median(v):7.2f} | max={v.max():8.2f} ms")
            if "curobo_time_ms" in df.columns:
                ik = pd.to_numeric(df.loc[t.index, "curobo_time_ms"], errors="coerce").fillna(0)
                rest = t["t_loop_total_ms"] - (t["t_read_pose_ms"] + ik + t["t_pb_query_ms"] + t["t_sleep_ms"])
                log(f"  {'unaccounted':11s}: mean={rest.mean():7.2f} | med={rest.median():7.2f} | max={rest.max():8.2f} ms")
    else:
        log("\n[WARN] Timer columns missing; skipping breakdown")

    log.save(csv_path.replace(".csv", "_freq_analysis.txt"))


def _safe(fn, *args):
    try:
        fn(*args)
    except Exception:
        print(f"[ERROR] {fn.__name__} failed:")
        traceback.print_exc()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/RT_20261021_172452_record.csv"
    base = re.sub(r"(_record|_vision|_control)?\.csv$", "", path)
    rec, vis, ctl = (f"{base}_{k}.csv" for k in ("record", "vision", "control"))
    has_rec, has_vis, has_ctl = (os.path.exists(p) for p in (rec, vis, ctl))

    if not (has_rec or has_vis or has_ctl):
        print(f"[ERROR] No record/vision/control CSV found for: {base}")
    if has_rec:
        _safe(analyze_record_frequency, rec)
    if has_vis:
        _safe(analyze_vision_frequency, vis)
    if has_rec and has_vis:
        _safe(correlate_long_frames, vis, rec)
    if has_ctl:
        _safe(analyze_frequency, ctl)