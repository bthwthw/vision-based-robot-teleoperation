import csv
import os
import time
 
class DataLogger:
    """
    Test filter:
    - raw vs filtered (position, quaternion) -> use cross-correlation to find delay 
    - jitter when freeze of raw vs filtered.
    """
 
    FIELDNAMES = [
        "wall_time_s", "frame_timestamp_s",
        "raw_x", "raw_y", "raw_z",
        "filt_x", "filt_y", "filt_z",
        "raw_qw", "raw_qx", "raw_qy", "raw_qz",
        "filt_qw", "filt_qx", "filt_qy", "filt_qz",
        "gripper_dist_mm",
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
            raw_quat=None, filt_quat=None, gripper_dist_mm=None):
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
 
        row["gripper_dist_mm"] = gripper_dist_mm
 
        self._writer.writerow(row)
 
    def close(self):
        self._file.close()
        print(f"[LOGGER INFO] Clossing log: {self.filepath}")