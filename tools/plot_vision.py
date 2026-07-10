"""
Cách dùng nhanh (đứng độc lập):
    python module_plot.py logs/RT_xxx.csv

Cách dùng trong code (main.py / analyze_log.py):
    from module_plot import generate_report_figures
    generate_report_figures("logs/RT_xxx.csv", out_dir="figs")
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,               # Kích thước chữ cơ sở
    "axes.titlesize": 14,          # Tiêu đề đồ thị
    "axes.labelsize": 12,          # Tên trục (X, Y)
    "xtick.labelsize": 11,         # Số trên trục X
    "ytick.labelsize": 11,         # Số trên trục Y
    "legend.fontsize": 11,         # Chú giải
    "figure.titlesize": 16,        # Tiêu đề tổng
    "axes.grid": True,             # Bật lưới
    "grid.alpha": 0.3,             # Độ mờ của lưới
    "grid.linestyle": "--"
})

def _fix_quaternion_continuity(q):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def _load(csv_path):
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    t0 = df["frame_timestamp_s"].iloc[0]
    df["t"] = df["frame_timestamp_s"] - t0  # thời gian tương đối (giây), dễ đọc trục hoành hơn

    raw_q = _fix_quaternion_continuity(df[["raw_qw", "raw_qx", "raw_qy", "raw_qz"]].to_numpy())
    filt_q = _fix_quaternion_continuity(df[["filt_qw", "filt_qx", "filt_qy", "filt_qz"]].to_numpy())
    ref = raw_q[0]
    df["raw_angle_deg"] = np.degrees(2 * np.arccos(np.clip(np.abs(raw_q @ ref), -1, 1)))
    df["filt_angle_deg"] = np.degrees(2 * np.arccos(np.clip(np.abs(filt_q @ ref), -1, 1)))

    dot = np.clip(np.sum(raw_q[:-1] * raw_q[1:], axis=1), -1, 1)
    step_deg = np.degrees(2 * np.arccos(np.abs(dot)))
    dt = np.diff(df["t"].to_numpy())
    dt[dt <= 0] = np.nan
    df["angvel_deg_s"] = np.concatenate([[0.0], step_deg / dt])

    return df

def add_subplot_label(ax, label):
    """Thêm nhãn (a), (b), (c) ở góc trên bên trái của subplot."""
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top')

# ---------- Các hàm vẽ riêng lẻ ----------

def plot_position_group(df, save_path):
    """Hình 1: Quỹ đạo tịnh tiến 3 trục X, Y, Z"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["(a)", "(b)", "(c)"]
    
    for i, axis in enumerate(["x", "y", "z"]):
        axes[i].plot(df["t"], df[f"raw_{axis}"] * 1000, color="tab:gray", alpha=0.7, lw=1.2, label="Raw Signal")
        axes[i].plot(df["t"], df[f"filt_{axis}"] * 1000, color="tab:blue", lw=1.8, label="1 Euro Filter")
        
        axes[i].set_ylabel(f"Position {axis.upper()} (mm)")
        axes[i].legend(loc="best")
        add_subplot_label(axes[i], labels[i])
        
    axes[-1].set_xlabel("Time (s)")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_orientation_group(df, save_path, reject_threshold_deg_s=860):
    """Hình 2: Phân tích Định hướng và Động học"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # (a) Biểu đồ lệch góc
    axes[0].plot(df["t"], df["raw_angle_deg"], color="tab:gray", alpha=0.7, lw=1.2, label="Raw Angle")
    axes[0].plot(df["t"], df["filt_angle_deg"], color="tab:orange", lw=1.8, label="Filtered Angle")
    axes[0].set_ylabel("Angular Div (deg)")
    axes[0].legend(loc="best")
    add_subplot_label(axes[0], "(a)")

    # (b) Vận tốc góc & Outlier
    axes[1].plot(df["t"], df["angvel_deg_s"], color="tab:purple", lw=1.2, label="Angular Velocity")
    axes[1].axhline(reject_threshold_deg_s, color="red", ls="--", lw=1.5, 
                    label=f"Outlier Threshold ({reject_threshold_deg_s} deg/s)")
    
    outliers = df[df["angvel_deg_s"] > reject_threshold_deg_s]
    if len(outliers) > 0:
        axes[1].scatter(outliers["t"], outliers["angvel_deg_s"], color="red", s=30, zorder=5, 
                        label=f"Outliers ({len(outliers)} frames)")
        
    axes[1].set_ylabel("Velocity (deg/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best")
    add_subplot_label(axes[1], "(b)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_evaluation_bar(df, save_path):
    """Hình 3: Biểu đồ cột tổng kết hiệu năng giảm nhiễu (Jitter Reduction)"""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels, reductions = [], []
    
    # Tính Jitter cho X, Y, Z
    for axis in ["x", "y", "z"]:
        jr = np.std(np.diff(df[f"raw_{axis}"])) * 1000
        jf = np.std(np.diff(df[f"filt_{axis}"])) * 1000
        reduction = 100 * (1 - jf / jr) if jr > 0 else 0
        labels.append(f"Axis {axis.upper()}")
        reductions.append(reduction)

    # Tính Jitter cho Orientation
    step_raw = df["raw_angle_deg"].diff().abs()
    step_filt = df["filt_angle_deg"].diff().abs()
    jr_o, jf_o = step_raw.std(), step_filt.std()
    reduction_o = 100 * (1 - jf_o / jr_o) if jr_o > 0 else 0
    labels.append("Orientation")
    reductions.append(reduction_o)

    colors = ["#3498db", "#3498db", "#3498db", "#e67e22"] # Xanh cho Pos, Cam cho Ori
    bars = ax.bar(labels, reductions, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    
    # Thêm số % lên đầu mỗi cột
    for b, v in zip(bars, reductions):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", 
                ha="center", fontweight='bold', fontsize=11)
                
    ax.set_ylabel("Jitter Reduction (%)")
    ax.set_ylim(0, max(reductions) + 15) 
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# main 

def generate_report_figures(csv_path, out_dir="figs"):
    """
    Tạo 3 hình ảnh chuẩn học thuật để chèn vào Word/LaTeX.
    """
    if not os.path.exists(csv_path):
        print(f"[PLOT ERR] File not found: {csv_path}")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    df = _load(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Đã sửa định dạng tên file: Đưa base_name lên đầu
    path_pos = os.path.join(out_dir, f"{base_name}_fig1_position.png")
    plot_position_group(df, path_pos)
    
    path_ori = os.path.join(out_dir, f"{base_name}_fig2_orientation.png")
    plot_orientation_group(df, path_ori)
    
    path_bar = os.path.join(out_dir, f"{base_name}_fig3_performance.png")
    plot_evaluation_bar(df, path_bar)

    print(f"\n[PLOT INFO] Đã xuất thành công 3 ảnh báo cáo (300 DPI) tại thư mục: {out_dir}/")
    print(f"  -> {os.path.basename(path_pos)}")
    print(f"  -> {os.path.basename(path_ori)}")
    print(f"  -> {os.path.basename(path_bar)}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/RT_20260710_112511.csv"
    generate_report_figures(path)