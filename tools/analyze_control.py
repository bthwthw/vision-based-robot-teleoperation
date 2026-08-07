import os
import sys
import numpy as np
import pandas as pd

def _fix_quaternion_continuity(q):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0: q[i] = -q[i]
    return q

def analyze_control(csv_path):
    if not os.path.exists(csv_path):
        alt_path = csv_path.replace(".csv", "_control.csv")
        if os.path.exists(alt_path):
            print(f"[CONTROL ANALYZER INFO] {csv_path} not found, using {alt_path} instead")
            csv_path = alt_path
        else:
            print(f"[CONTROL ANALYZER ERROR] File not found: {csv_path}")
            return

    try:
        df = pd.read_csv(csv_path).dropna(subset=["curobo_time_ms", "ik_success"]).reset_index(drop=True)
        if len(df) < 5:
            print("[CONTROL ANALYZER WARN] Empty CSV")
            return

        report_lines = []
        def log_print(msg=""):
            print(msg)
            report_lines.append(msg)

        log_print(f"\n=== CUROBO & PYBULLET CONTROL ANALYSIS ===")
        log_print(f"File   : {os.path.basename(csv_path)}")
        
        # 1. Hiệu năng tính toán cuRobo IK
        latency = df["curobo_time_ms"].to_numpy()
        ik_success_arr = df["ik_success"].to_numpy()
        sr = (np.sum(ik_success_arr) / len(ik_success_arr)) * 100
        
        log_print(f"\n[IK SOLVER PERFORMANCE]")
        log_print(f"  IK Solvability Rate: {sr:.1f}%")
        log_print(f"  cuRobo GPU Latency: mean={np.mean(latency):.2f}ms | max={np.max(latency):.2f}ms | std={np.std(latency):.2f}ms")

        # 2. Sai số bám vị trí (Euclidean Position Error)
        log_print("\n[POSITION TRACKING ERROR (Tgt vs PyBullet Actual)]")
        err_x = df["tgt_tcp_x"] - df["pb_tcp_x"]
        err_y = df["tgt_tcp_y"] - df["pb_tcp_y"]
        err_z = df["tgt_tcp_z"] - df["pb_tcp_z"]
        euclidean_err_mm = np.sqrt(err_x**2 + err_y**2 + err_z**2) * 1000.0
        
        log_print(f"  Max Euclidean Error : {np.max(euclidean_err_mm):.2f} mm")
        log_print(f"  Mean Euclidean Error: {np.mean(euclidean_err_mm):.2f} mm")
        log_print(f"  Std Euclidean Error : {np.std(euclidean_err_mm):.2f} mm")

        # 3. Sai số bám hướng (Orientation Angle Deviation)
        log_print("\n[ORIENTATION TRACKING ERROR]")
        tgt_q = _fix_quaternion_continuity(df[["tgt_tcp_qw", "tgt_tcp_qx", "tgt_tcp_qy", "tgt_tcp_qz"]].to_numpy())
        pb_q = _fix_quaternion_continuity(df[["pb_tcp_qw", "pb_tcp_qx", "pb_tcp_qy", "pb_tcp_qz"]].to_numpy())
        
        dot_product = np.clip(np.sum(tgt_q * pb_q, axis=1), -1.0, 1.0)
        ang_err_deg = np.degrees(2 * np.arccos(np.abs(dot_product)))
        log_print(f"  Mean Orientation Error: {np.mean(ang_err_deg):.2f} deg")
        log_print(f"  Max Orientation Error : {np.max(ang_err_deg):.2f} deg")

        # 4. Kiểm định động học vận tốc khớp (Kinematic Bounds Check)
        log_print("\n[JOINT KINEMATIC CHECK (PyBullet Feedback)]")
        t = df["frame_timestamp_s"].to_numpy()
        dt = np.diff(t)
        dt[dt <= 0] = 1e-4  # tránh chia cho 0
        
        max_omega_joints = []
        for idx in range(1, 7):
            q_act = df[f"pb_q_{idx}"].to_numpy()
            dq_act = np.abs(np.diff(q_act)) / dt
            max_omega_joints.append(np.max(dq_act))
            log_print(f"  Joint {idx} Max Velocity: {np.max(dq_act):.2f} rad/s")
            
        if np.max(max_omega_joints) > 5.0:
            log_print("  [WARNING] Omega >5.0 rad/s, maybe singularity")
        else:
            log_print("  [SUCCESS] Omega okay")

        # Xuất file báo cáo dạng text
        full_report = "\n".join(report_lines)
        txt_path = csv_path.replace(".csv", "_control_analysis.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n[ANALYZER INFO] Saved at: {txt_path}\n")
        
    except Exception as e:
        print(f"[CONTROL ANALYZER ERROR] Analysis failed: {e}\n")

if __name__ == "__main__":
    default_path = os.path.join("logs", "RT_20260729_132013_control.csv")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    analyze_control(path)