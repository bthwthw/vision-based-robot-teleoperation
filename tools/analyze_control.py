import os
import sys
import traceback
import numpy as np
import pandas as pd

JOINT_VEL_REF_RAD_S = 5.0
N_JOINTS = 6
TIME_COL = "wall_time_s"
SEG_GAP_FACTOR = 3.0
MAX_DELAY_TICKS = 3


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


def _num(df, cols):
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _bool_col(s):
    m = {"true": 1.0, "1": 1.0, "1.0": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0}
    return s.astype(str).str.strip().str.lower().map(m)


def _stats(x, unit, digits=2):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "n/a"
    f = f".{digits}f"
    return (f"mean={x.mean():{f}}{unit} | med={np.median(x):{f}}{unit} | "
            f"p95={np.percentile(x, 95):{f}}{unit} | max={x.max():{f}}{unit}")


def _segments(df):
    # New segment after a long gap (log only runs while teleop is active)
    t = _num(df, [TIME_COL])[TIME_COL]
    dt = t.diff()
    thr = SEG_GAP_FACTOR * dt[dt > 0].median()
    seg = (dt.isna() | (dt > thr) | (dt <= 0)).cumsum()
    return seg, thr


def _errors(df, seg, k):
    # Compare target[i] with PyBullet[i+k] inside the same segment
    tp = [f"tgt_tcp_{a}" for a in "xyz"]
    tq = [f"tgt_tcp_q{c}" for c in "wxyz"]
    pp = [f"pb_tcp_{a}" for a in "xyz"]
    pq = [f"pb_tcp_q{c}" for c in "wxyz"]
    d = _num(df, tp + tq + pp + pq)
    T = d[tp + tq].to_numpy()
    P = d[pp + pq].shift(-k).to_numpy()
    same = seg.eq(seg.shift(-k)).to_numpy()
    ok = same & np.isfinite(T).all(axis=1) & np.isfinite(P).all(axis=1)
    T, P = T[ok], P[ok]

    pos = np.linalg.norm(T[:, :3] - P[:, :3], axis=1) * 1000.0
    qt, qp = T[:, 3:], P[:, 3:]
    nt, npb = np.linalg.norm(qt, axis=1), np.linalg.norm(qp, axis=1)
    good = (nt > 1e-9) & (npb > 1e-9)
    dot = np.clip(np.abs(np.sum(qt * qp, axis=1) / np.where(good, nt * npb, 1.0)), 0.0, 1.0)
    ori = np.where(good, np.degrees(2.0 * np.arccos(dot)), np.nan)
    return pd.DataFrame({"pos": pos, "ori": ori}, index=np.where(ok)[0])


def analyze_control(csv_path):
    if not os.path.exists(csv_path):
        alt_path = csv_path.replace(".csv", "_control.csv")
        if os.path.exists(alt_path):
            print(f"[INFO] {csv_path} not found, using {alt_path}")
            csv_path = alt_path
        else:
            print(f"[ERROR] File not found: {csv_path}")
            return

    try:
        df = pd.read_csv(csv_path)
        ik = _bool_col(df["ik_success"])
        lat = _num(df, ["curobo_time_ms"])["curobo_time_ms"]
        ok = ik.notna() & lat.notna()
        if ok.sum() < 5:
            print("[WARN] Fewer than 5 valid rows. Skipping.")
            return

        seg, gap_thr = _segments(df)
        t = _num(df, [TIME_COL])[TIME_COL]
        dt = t.diff()
        active_s = dt[seg.eq(seg.shift()) & (dt > 0)].sum()

        log = Report()
        log("\n=== CONTROL ANALYSIS ===")
        log(f"File : {os.path.basename(csv_path)}")
        log(f"Rows : {len(df)} (valid IK rows: {int(ok.sum())})")
        log(f"Active segments: {seg.nunique()} (gap > {gap_thr * 1000:.0f}ms) | active time: {active_s:.1f}s")

        log("\n[IK SOLVER]")
        log(f"  Success rate: {ik[ok].mean() * 100:.1f}%")
        log(f"  Latency: {_stats(lat[ok], 'ms')} | std={lat[ok].std():.2f}ms")

        errs = {k: _errors(df, seg, k) for k in range(MAX_DELAY_TICKS + 1)}
        best_k = min(errs, key=lambda k: errs[k]["pos"].mean() if len(errs[k]) else np.inf)

        log("\n[TRACKING ERROR] target[i] vs PyBullet[i+k]")
        for k in sorted({0, best_k}):
            tag = "same row" if k == 0 else "best delay"
            log(f"  k={k} ({tag})")
            log(f"    pos: {_stats(errs[k]['pos'], 'mm')}")
            log(f"    ori: {_stats(errs[k]['ori'], 'deg')}")
        if best_k > 0:
            log(f"  Same-row error is mostly delay: PyBullet reaches the target ~{best_k} tick(s) later")

        if (ik[ok] == 0).any():
            e = errs[best_k]
            fail_idx = ik.index[ik == 0.0]
            e_ok = e.loc[e.index.difference(fail_idx)]
            log(f"  IK success only (k={best_k}): pos {_stats(e_ok['pos'], 'mm')}")

        log("\n[JOINT VELOCITY (PyBullet feedback, per control tick)]")
        consecutive = seg.eq(seg.shift()) & (dt > 0)
        worst, n_over = 0.0, 0
        for i in range(1, N_JOINTS + 1):
            q = _num(df, [f"pb_q_{i}"])[f"pb_q_{i}"]
            dq = q.diff().abs()
            m = consecutive & dq.notna()
            v = (dq[m] / dt[m]).to_numpy()
            if v.size == 0:
                log(f"  J{i}: n/a")
                continue
            worst = max(worst, v.max())
            n_over += int((v > JOINT_VEL_REF_RAD_S).sum())
            log(f"  J{i}: max={v.max():.2f} | p99={np.percentile(v, 99):.2f} rad/s")
        log(f"  Pairs skipped (gaps/invalid dt): {int((~consecutive).sum()) - 1}")

        if worst > JOINT_VEL_REF_RAD_S:
            log(f"  [WARN] {n_over} samples > {JOINT_VEL_REF_RAD_S} rad/s (ref limit); check singularity/timestamps")
        else:
            log(f"  [OK] Max velocity below {JOINT_VEL_REF_RAD_S} rad/s (ref limit)")

        log.save(csv_path.replace(".csv", "_control_analysis.txt"))

    except Exception:
        print("[ERROR] Analysis failed:")
        traceback.print_exc()


if __name__ == "__main__":
    default_path = os.path.join("logs", "RT_20261021_172452_control.csv")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    analyze_control(path)