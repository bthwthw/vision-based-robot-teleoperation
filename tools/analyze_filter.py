import sys
import os
import traceback
import numpy as np
import pandas as pd

TIME_COL = "frame_timestamp_s"
MAX_LAG_S = 0.5
OMEGA_REF_DEG_S = 1000.0
GAP_FACTOR = 3.0
FLAT_PEAK_DROP = 0.002
MIN_PEAK_CORR = 0.8
DESPIKE_WIN = 5


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


def _series(df, cols):
    d = df[[TIME_COL] + cols].apply(pd.to_numeric, errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values(TIME_COL)
    return d[~d[TIME_COL].duplicated()].reset_index(drop=True)


def _unit_quats(d, cols):
    q = d[cols].to_numpy()
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def estimate_lag(t, raw, filt, max_lag_s=MAX_LAG_S):
    # Returns (lag_s, peak_corr, reliable); lag > 0 means filt lags raw
    t = np.asarray(t, dtype=float)
    if len(t) < 10:
        return np.nan, np.nan, False
    step = np.median(np.diff(t))
    if step <= 0:
        return np.nan, np.nan, False

    # Resample to a uniform grid so lag in samples maps to seconds
    grid = np.arange(t[0], t[-1], step)
    x = np.interp(grid, t, raw)
    y = np.interp(grid, t, filt)
    n = len(grid)

    # Mask grid points that fall inside long gaps
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    ok = (t[idx] - t[idx - 1]) <= GAP_FACTOR * step

    # Pearson correlation per lag over valid overlap
    max_k = min(int(max_lag_s / step), n // 2)
    ks = np.arange(-max_k, max_k + 1)
    corr = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        if k >= 0:
            a_, b_, m = x[:n - k], y[k:], ok[:n - k] & ok[k:]
        else:
            a_, b_, m = x[-k:], y[:n + k], ok[-k:] & ok[:n + k]
        if m.sum() < 30 or a_[m].std() < 1e-12 or b_[m].std() < 1e-12:
            continue
        corr[i] = np.corrcoef(a_[m], b_[m])[0, 1]
    if np.all(np.isnan(corr)):
        return np.nan, np.nan, False

    i = int(np.nanargmax(corr))
    off = 0.0
    if 0 < i < len(ks) - 1 and np.isfinite(corr[i - 1]) and np.isfinite(corr[i + 1]):
        den = corr[i - 1] - 2 * corr[i] + corr[i + 1]
        if den < 0:
            # Parabolic sub-sample refinement
            off = 0.5 * (corr[i - 1] - corr[i + 1]) / den

    # Unreliable if peak is at the search edge, flat (lags 3 steps away almost as good), or low correlation
    nb = [corr[j] for j in (i - 3, i + 3) if 0 <= j < len(ks) and np.isfinite(corr[j])]
    reliable = (0 < i < len(ks) - 1 and bool(nb) and (corr[i] - max(nb)) > FLAT_PEAK_DROP
                and corr[i] > MIN_PEAK_CORR)
    return (ks[i] + off) * step, corr[i], reliable


def _lag_str(res):
    lag, r, ok = res
    if not np.isfinite(lag):
        return "n/a"
    flag = "" if ok else ", unreliable"
    return f"{lag * 1000:.0f} ms (r={r:.3f}{flag})"


def _despike(x, win=DESPIKE_WIN):
    # Rolling median to suppress isolated spikes
    h = win // 2
    p = np.pad(np.asarray(x, dtype=float), h, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, win), axis=1)


def _jitter(x, order=2):
    x = np.asarray(x, dtype=float)
    if len(x) <= order:
        return np.nan
    return np.mean(np.abs(np.diff(x, n=order)))


def _reduction(jr, jf):
    return 100.0 * (1.0 - jf / jr) if np.isfinite(jr) and jr > 0 else np.nan


def _jitter_line(log, name, jr, jf, unit):
    red = _reduction(jr, jf)
    red_s = "n/a" if not np.isfinite(red) else f"{red:.0f}%"
    log(f"  {name:5s}: raw={jr:.3f}{unit} | filt={jf:.3f}{unit} | reduction={red_s}")


def analyze_vision(csv_path):
    if not os.path.exists(csv_path):
        alt_path = csv_path.replace(".csv", "_vision.csv")
        if os.path.exists(alt_path):
            print(f"[INFO] {csv_path} not found, using {alt_path}")
            csv_path = alt_path
        else:
            print(f"[ERROR] File not found: {csv_path}")
            return

    try:
        df = pd.read_csv(csv_path)
        t_all = pd.to_numeric(df[TIME_COL], errors="coerce").dropna().to_numpy()
        dt_all = np.diff(t_all)
        dt_pos = dt_all[dt_all > 0]
        if len(dt_pos) < 10:
            print("[WARN] Fewer than 10 valid frames. Skipping.")
            return

        log = Report()
        log("\n=== FILTER ANALYSIS ===")
        log(f"File  : {os.path.basename(csv_path)}")
        log(f"Frames: {len(df)} | rows with NaN: {int(df.isna().any(axis=1).sum())} | "
            f"non-positive dt: {int((dt_all <= 0).sum())}")
        dt_ms = dt_pos * 1000.0
        log(f"Frame dt: mean={dt_ms.mean():.1f} | med={np.median(dt_ms):.1f} | "
            f"p95={np.percentile(dt_ms, 95):.1f} | max={dt_ms.max():.1f} ms | "
            f"rate={len(dt_pos) / dt_pos.sum():.1f} Hz")
        n_gap = int((dt_pos > GAP_FACTOR * np.median(dt_pos)).sum())
        if n_gap:
            log(f"[INFO] {n_gap} gaps > {GAP_FACTOR * np.median(dt_pos) * 1000:.0f}ms "
                f"({GAP_FACTOR:.0f}x median); excluded from lag")

        # Lag
        log("\n[LAG] (filt vs raw, + = filt lags)")
        for axis in "xyz":
            d = _series(df, [f"raw_{axis}", f"filt_{axis}"])
            lag = estimate_lag(d[TIME_COL], d[f"raw_{axis}"], d[f"filt_{axis}"])
            log(f"  {axis.upper()}   : {_lag_str(lag)}")

        d = _series(df, ["raw_gripper_dist_mm", "filt_gripper_dist_mm"])
        lag = estimate_lag(d[TIME_COL], d["raw_gripper_dist_mm"], d["filt_gripper_dist_mm"])
        log(f"  Grip: {_lag_str(lag)}")

        raw_c = [f"raw_q{c}" for c in "wxyz"]
        filt_c = [f"filt_q{c}" for c in "wxyz"]
        dq = _series(df, raw_c + filt_c)
        nonzero = (np.linalg.norm(dq[raw_c].to_numpy(), axis=1) > 1e-9) & \
                  (np.linalg.norm(dq[filt_c].to_numpy(), axis=1) > 1e-9)
        dq = dq[nonzero].reset_index(drop=True)
        raw_q = _unit_quats(dq, raw_c)
        filt_q = _unit_quats(dq, filt_c)

        ref = raw_q[0]
        ang_raw = 2 * np.arccos(np.clip(np.abs(raw_q @ ref), 0.0, 1.0))
        ang_filt = 2 * np.arccos(np.clip(np.abs(filt_q @ ref), 0.0, 1.0))
        lag = estimate_lag(dq[TIME_COL], _despike(ang_raw), ang_filt)
        log(f"  Ori : {_lag_str(lag)} (raw despiked)")

        # Jitter
        log("\n[JITTER] (mean |2nd diff|, lower = smoother)")
        for axis in "xyz":
            d = _series(df, [f"raw_{axis}", f"filt_{axis}"])
            jr = _jitter(d[f"raw_{axis}"]) * 1000
            jf = _jitter(d[f"filt_{axis}"]) * 1000
            _jitter_line(log, axis.upper(), jr, jf, "mm")

        d = _series(df, ["raw_gripper_dist_mm", "filt_gripper_dist_mm"])
        _jitter_line(log, "Grip", _jitter(d["raw_gripper_dist_mm"]),
                     _jitter(d["filt_gripper_dist_mm"]), "mm")

        step_raw = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(raw_q[:-1] * raw_q[1:], axis=1)), 0.0, 1.0)))
        step_filt = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(filt_q[:-1] * filt_q[1:], axis=1)), 0.0, 1.0)))
        _jitter_line(log, "Ori", _jitter(step_raw, 1), _jitter(step_filt, 1), "deg")

        # Angular velocity
        dt = np.diff(dq[TIME_COL].to_numpy())
        valid = dt > 0
        om_raw = step_raw[valid] / dt[valid]
        om_filt = step_filt[valid] / dt[valid]

        log("\n[ANGULAR VELOCITY]")
        log(f"  Raw : max={om_raw.max():.0f} | p99={np.percentile(om_raw, 99):.0f} | mean={om_raw.mean():.1f} deg/s")
        log(f"  Filt: max={om_filt.max():.0f} | p99={np.percentile(om_filt, 99):.0f} | mean={om_filt.mean():.1f} deg/s")
        log(f"  Dropped non-positive dt pairs: {int((~valid).sum())}")

        n_raw = int((om_raw > OMEGA_REF_DEG_S).sum())
        n_filt = int((om_filt > OMEGA_REF_DEG_S).sum())
        if n_raw:
            log(f"  [INFO] Raw: {n_raw} samples > {OMEGA_REF_DEG_S:.0f} deg/s")
        if n_filt:
            log(f"  [WARN] Filt: {n_filt} samples > {OMEGA_REF_DEG_S:.0f} deg/s (ref); check slew limiter")
        else:
            log(f"  [OK] Filt: no sample > {OMEGA_REF_DEG_S:.0f} deg/s (ref, not robot limit)")

        log.save(csv_path.replace(".csv", "_analysis.txt"))

    except Exception:
        print("[ERROR] Analysis failed:")
        traceback.print_exc()


if __name__ == "__main__":
    default_path = os.path.join("logs", "RT_20261021_172452_vision.csv")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    analyze_vision(path)