import csv
import os
import time
 
class DataLogger:
 
    FIELDNAMES = [
        "wall_time_s", "frame_timestamp_s",
        "raw_x", "raw_y", "raw_z",
        "filt_x", "filt_y", "filt_z",
        "raw_qw", "raw_qx", "raw_qy", "raw_qz",
        "filt_qw", "filt_qx", "filt_qy", "filt_qz",
        "raw_gripper_dist_mm", "filt_gripper_dist_mm",

        "curobo_time_ms", "ik_success",
        "tgt_tcp_x", "tgt_tcp_y", "tgt_tcp_z",
        "tgt_tcp_qw", "tgt_tcp_qx", "tgt_tcp_qy", "tgt_tcp_qz",
        "pb_tcp_x", "pb_tcp_y", "pb_tcp_z",
        "pb_tcp_qw", "pb_tcp_qx", "pb_tcp_qy", "pb_tcp_qz",
        "q_tgt_1", "q_tgt_2", "q_tgt_3", "q_tgt_4", "q_tgt_5", "q_tgt_6",
        "pb_q_1", "pb_q_2", "pb_q_3", "pb_q_4", "pb_q_5", "pb_q_6"
    ]
 
    def __init__(self, filepath="log.csv", out_dir="logs"):
        os.makedirs(out_dir, exist_ok=True)
        self.filepath = os.path.join(out_dir, filepath)
        self._file = open(self.filepath, mode="w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._start_wall_time = time.time()
        print(f"[LOGGER INFO] Logging into: {self.filepath}")
 
    def log(self, frame_timestamp_s, raw_pos=None, filt_pos=None,
            raw_quat=None, filt_quat=None, raw_gripper_dist=None, filt_gripper_dist=None,
            curobo_time_ms=None, ik_success=None, tgt_tcp_pos=None, tgt_tcp_quat=None,
            pb_tcp_pos=None, pb_tcp_quat=None, q_tgt=None, pb_q=None):
        
        row = {
            "wall_time_s": round(time.time() - self._start_wall_time, 6),
            "frame_timestamp_s": frame_timestamp_s,
        }
 
        row["raw_x"], row["raw_y"], row["raw_z"] = raw_pos if raw_pos else (None, None, None)
        row["filt_x"], row["filt_y"], row["filt_z"] = filt_pos if filt_pos else (None, None, None)
        
        if raw_quat is not None:
            row["raw_qw"], row["raw_qx"], row["raw_qy"], row["raw_qz"] = raw_quat
        else:
            row["raw_qw"] = row["raw_qx"] = row["raw_qy"] = row["raw_qz"] = None
 
        if filt_quat is not None:
            row["filt_qw"], row["filt_qx"], row["filt_qy"], row["filt_qz"] = filt_quat
        else:
            row["filt_qw"] = row["filt_qx"] = row["filt_qy"] = row["filt_qz"] = None
 
        row["raw_gripper_dist_mm"] = raw_gripper_dist if raw_gripper_dist is not None else None
        row["filt_gripper_dist_mm"] = filt_gripper_dist if filt_gripper_dist is not None else None

        row["curobo_time_ms"] = curobo_time_ms
        row["ik_success"] = ik_success
        
        if tgt_tcp_pos is not None:
            row["tgt_tcp_x"], row["tgt_tcp_y"], row["tgt_tcp_z"] = tgt_tcp_pos
        if tgt_tcp_quat is not None:
            row["tgt_tcp_qw"], row["tgt_tcp_qx"], row["tgt_tcp_qy"], row["tgt_tcp_qz"] = tgt_tcp_quat
            
        if pb_tcp_pos is not None:
            row["pb_tcp_x"], row["pb_tcp_y"], row["pb_tcp_z"] = pb_tcp_pos
        if pb_tcp_quat is not None:
            row["pb_tcp_qw"], row["pb_tcp_qx"], row["pb_tcp_qy"], row["pb_tcp_qz"] = pb_tcp_quat
            
        if q_tgt is not None and len(q_tgt) >= 6:
            for idx in range(6): row[f"q_tgt_{idx+1}"] = q_tgt[idx]
        if pb_q is not None and len(pb_q) >= 6:
            for idx in range(6): row[f"pb_q_{idx+1}"] = pb_q[idx]

        self._writer.writerow(row)
 
    def close(self):
        self._file.close()
        print(f"[LOGGER INFO] Clossing log: {self.filepath}")