import sys
import os
import traceback
import numpy as np
import pandas as pd

TIME_COL = "frame_timestamp_s"
ROT_GESTURE = {1: "Pointing_Up", -1: "Victory"}
GRIP_GESTURE = {1: "Closed_Fist", 0: "Open_Palm"}


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


def analyze_gesture(csv_path, min_score=0.7, pre=6, post=2):
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    try:
        # keep "None" as a label, empty cell = no hand
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
        t = df[TIME_COL].to_numpy()
        w = df["wall_time_s"].to_numpy()
        raw = df["gesture_raw"].fillna("-").to_numpy()
        score = df["gesture_score"].fillna(0).to_numpy()
        rc = df["rot_cmd"].fillna(0).astype(int).to_numpy()
        gs = df["gripper_state"].fillna(0).astype(int).to_numpy()
        act = df["teleop_active"].fillna(0).astype(int).to_numpy() if "teleop_active" in df.columns else None

        log = Report()
        log("\n=== GESTURE ANALYSIS ===")
        log(f"File  : {os.path.basename(csv_path)}")
        dt = np.diff(t)
        dt = dt[dt > 0] * 1000.0
        log(f"Frames: {len(df)} | dt med={np.median(dt):.0f} | p95={np.percentile(dt, 95):.0f} ms | rate={1000.0 / dt.mean():.1f} Hz")
        log(f"teleop_active column: {'yes' if act is not None else 'no (space releases are guessed)'}")

        log(f"\n[SCORES] (top-1 score per label, threshold ref = {min_score})")
        for g in ("Open_Palm", "Closed_Fist", "Pointing_Up", "Victory"):
            s = score[raw == g]
            if len(s) == 0:
                continue
            log(f"  {g:12s}: n={len(s):4d} | p5={np.percentile(s, 5):.2f} | p50={np.percentile(s, 50):.2f} | below ref={np.mean(s < min_score):.0%}")

        stops = [i for i in range(1, len(df)) if rc[i - 1] != 0 and rc[i] == 0]
        lags = []
        log(f"\n[ROT STOPS] total={len(stops)}")
        for i in stops:
            g = ROT_GESTURE[rc[i - 1]]
            s = i - 1
            while s > 0 and rc[s - 1] == rc[i - 1]:
                s -= 1

            valid = [k for k in range(s, i) if raw[k] == g and score[k] >= min_score]
            last_ok = valid[-1] if valid else s
            lag_ms = (t[i] - t[last_ok + 1]) * 1000 if last_ok + 1 < i else 0.0

            if act is not None:
                release = act[i - 1] == 1 and act[i] == 0
            else:
                release = lag_ms == 0.0
            if not release:
                lags.append(lag_ms)

            tag = "space release" if release else f"invalid->stop={lag_ms:.0f}ms"
            log(f"\n  t={w[i]:.2f}s run={w[i - 1] - w[s]:.2f}s ({g}) {tag}")
            for k in range(max(s, i - pre), min(len(df), i + post + 1)):
                log(f"    {w[k]:.3f}  {raw[k]:<12} {score[k]:.2f}  rot={rc[k]:+d}")
        if lags:
            log(f"\n  gesture-change stops: n={len(lags)} | median={np.median(lags):.0f} | max={np.max(lags):.0f} ms")

        chg = [i for i in range(1, len(df)) if gs[i] != gs[i - 1]]
        delays = []
        log(f"\n[GRIPPER] latch changes={len(chg)}")
        for i in chg:
            tgt = GRIP_GESTURE[gs[i]]
            k = i
            while k > 0 and raw[k - 1] == tgt:
                k -= 1
            if act is not None:
                while k < i and act[k] == 0:
                    k += 1
            run_s = score[k:i + 1]
            delay_ms = (t[i] - t[k]) * 1000
            delays.append(delay_ms)
            log(f"  t={w[i]:.2f}s -> {tgt:11s} delay={delay_ms:5.0f}ms frames={i - k + 1:3d} "
                f"below ref={int((run_s < min_score).sum()):3d} min={run_s.min():.2f} first={run_s[0]:.2f}")
        if delays:
            log(f"\n  delay: median={np.median(delays):.0f} | max={np.max(delays):.0f} ms")

        log.save(csv_path.replace(".csv", "_gesture.txt"))

    except Exception:
        print("[ERROR] Analysis failed:")
        traceback.print_exc()


if __name__ == "__main__":
    default_path = os.path.join("logs", "RT_20261028_134249_vision.csv")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    analyze_gesture(path)