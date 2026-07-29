import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"
})

def generate_control_figures(csv_path, out_dir="figs"):
    if not os.path.exists(csv_path):
        return
        
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna(subset=["curobo_time_ms", "ik_success"]).reset_index(drop=True)
    if df.empty or len(df) < 2: return
    
    t0 = df["frame_timestamp_s"].iloc[0]
    t = df["frame_timestamp_s"] - t0
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # --- HÌNH 1: QUỸ ĐẠO BÁM TRỤC TCP POSITION (CUROBO VS PYBULLET) ---
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = ["tab:red", "tab:green", "tab:blue"]
    for i, axis in enumerate(["x", "y", "z"]):
        axes1[i].plot(t, df[f"tgt_tcp_{axis}"] * 1000, color="black", ls="--", alpha=0.8, lw=1.5, label="cuRobo Command")
        axes1[i].plot(t, df[f"pb_tcp_{axis}"] * 1000, color=colors[i], lw=2.0, label="PyBullet Actual")
        axes1[i].set_ylabel(f"TCP {axis.upper()} (mm)")
        axes1[i].legend(loc="upper right")
    axes1[-1].set_xlabel("Time (s)")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, f"{base_name}_ctrl_fig1_tracking.png"), dpi=300)
    plt.close(fig1)

    # --- HÌNH 2: SAI SỐ TỊNH TIẾN EUCLID VÀ TRẠNG THÁI HỘI TỤ ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    err_x = (df["tgt_tcp_x"] - df["pb_tcp_x"]) * 1000.0
    err_y = (df["tgt_tcp_y"] - df["pb_tcp_y"]) * 1000.0
    err_z = (df["tgt_tcp_z"] - df["pb_tcp_z"]) * 1000.0
    euclidean_error = np.sqrt(err_x**2 + err_y**2 + err_z**2)
    
    ax2.plot(t, euclidean_error, color="tab:red", lw=1.8, label="Euclidean Tracking Error")
    # Biểu diễn những vùng giải IK thất bại dưới dạng vạch đỏ nền
    fail_idx = df[df["ik_success"] == 0]
    if not fail_idx.empty:
        for ft in (fail_idx["frame_timestamp_s"] - t0):
            ax2.axvline(ft, color="red", alpha=0.1, zorder=0)
        ax2.plot([], [], color="red", alpha=0.3, label="IK Fail Region")
        
    ax2.set_ylabel("Error (mm)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"{base_name}_ctrl_fig2_error.png"), dpi=300)
    plt.close(fig2)

    # --- HÌNH 3: SỰ THAY ĐỔI CỦA 6 GÓC KHỚP THEO THỜI GIAN ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for idx in range(1, 7):
        ax3.plot(t, df[f"pb_q_{idx}"], lw=1.5, label=f"Joint {idx}")
    ax3.set_ylabel("Joint Position (rad)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, f"{base_name}_ctrl_fig3_joints.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)

    print(f"[CONTROL PLOT INFO] 3 figs saved at: {out_dir}/")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs\RT_20260729_132013.csv"
    generate_control_figures(path)