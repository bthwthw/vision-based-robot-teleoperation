import csv
import os
import time
import threading
import queue
 
 
class DataLogger:
    VISION_FIELDNAMES = [
        "wall_time_s", "frame_timestamp_s",
        "raw_x", "raw_y", "raw_z", "filt_x", "filt_y", "filt_z",
        "raw_qw", "raw_qx", "raw_qy", "raw_qz",
        "filt_qw", "filt_qx", "filt_qy", "filt_qz",
        "raw_gripper_dist_mm", "filt_gripper_dist_mm",
    ]
 
    CONTROL_FIELDNAMES = [
        "wall_time_s", "frame_timestamp_s",
        "curobo_time_ms", "ik_success",
        "tgt_tcp_x", "tgt_tcp_y", "tgt_tcp_z",
        "tgt_tcp_qw", "tgt_tcp_qx", "tgt_tcp_qy", "tgt_tcp_qz",
        "pb_tcp_x", "pb_tcp_y", "pb_tcp_z",
        "pb_tcp_qw", "pb_tcp_qx", "pb_tcp_qy", "pb_tcp_qz",
        "q_tgt_1", "q_tgt_2", "q_tgt_3", "q_tgt_4", "q_tgt_5", "q_tgt_6",
        "pb_q_1", "pb_q_2", "pb_q_3", "pb_q_4", "pb_q_5", "pb_q_6",
        "t_read_pose_ms", "t_pb_query_ms", "t_log_enqueue_ms",
        "t_sleep_ms", "t_loop_total_ms",
    ]
 
    _QUEUE_SENTINEL = object()
 
    def __init__(self, filename_prefix, out_dir="logs", flush_every_n=50):
        os.makedirs(out_dir, exist_ok=True)
        self._start_wall_time = time.time()
 
        if filename_prefix.endswith(".csv"):
            filename_prefix = filename_prefix[:-4]
 
        self.vision_path = os.path.join(out_dir, f"{filename_prefix}_vision.csv")
        self.control_path = os.path.join(out_dir, f"{filename_prefix}_control.csv")
 
        self._vision_file = open(self.vision_path, mode="w", newline="", buffering=1 << 16)
        self._control_file = open(self.control_path, mode="w", newline="", buffering=1 << 16)
        self._vision_writer = csv.DictWriter(self._vision_file, fieldnames=self.VISION_FIELDNAMES)
        self._control_writer = csv.DictWriter(self._control_file, fieldnames=self.CONTROL_FIELDNAMES)
        self._vision_writer.writeheader()
        self._control_writer.writeheader()
 
        self._flush_every_n = flush_every_n
        self._vision_count = 0
        self._control_count = 0
 
        self._queue = queue.Queue()
        self._closed = False
        self._worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker_thread.start()
 
        print(f"[LOGGER INFO] Vision  log -> {self.vision_path}")
        print(f"[LOGGER INFO] Control log -> {self.control_path}")
 
    def _writer_loop(self):
        while True:
            item = self._queue.get()
            if item is self._QUEUE_SENTINEL:
                self._queue.task_done()
                break
            kind, row = item
            try:
                if kind == "vision":
                    self._vision_writer.writerow(row)
                    self._vision_count += 1
                    if self._vision_count % self._flush_every_n == 0:
                        self._vision_file.flush()
                else:
                    self._control_writer.writerow(row)
                    self._control_count += 1
                    if self._control_count % self._flush_every_n == 0:
                        self._control_file.flush()
            except Exception as e:
                print(f"[LOGGER ERROR] Ghi log thất bại ({kind}): {e}")
            self._queue.task_done()
 
    def write_vision(self, data):
        row = {
            "wall_time_s": round(time.time() - self._start_wall_time, 6),
            "frame_timestamp_s": data.get("frame_timestamp_s"),
        }
 
        raw_pos = data.get("raw_pos")
        filt_pos = data.get("filt_pos")
        row["raw_x"], row["raw_y"], row["raw_z"] = raw_pos if raw_pos else (None, None, None)
        row["filt_x"], row["filt_y"], row["filt_z"] = filt_pos if filt_pos else (None, None, None)
 
        raw_quat = data.get("raw_quat")
        if raw_quat is not None:
            row["raw_qw"], row["raw_qx"], row["raw_qy"], row["raw_qz"] = raw_quat
        filt_quat = data.get("filt_quat")
        if filt_quat is not None:
            row["filt_qw"], row["filt_qx"], row["filt_qy"], row["filt_qz"] = filt_quat
 
        row["raw_gripper_dist_mm"] = data.get("raw_gripper_dist")
        row["filt_gripper_dist_mm"] = data.get("filt_gripper_dist")
 
        self._queue.put_nowait(("vision", row))
 
    def write_control(self, data):
        row = {
            "wall_time_s": round(time.time() - self._start_wall_time, 6),
            "frame_timestamp_s": data.get("frame_timestamp_s"),
            "curobo_time_ms": data.get("curobo_time_ms"),
            "ik_success": data.get("ik_success"),
        }
 
        tgt_tcp_pos = data.get("tgt_tcp_pos")
        if tgt_tcp_pos is not None:
            row["tgt_tcp_x"], row["tgt_tcp_y"], row["tgt_tcp_z"] = tgt_tcp_pos
        tgt_tcp_quat = data.get("tgt_tcp_quat")
        if tgt_tcp_quat is not None:
            row["tgt_tcp_qw"], row["tgt_tcp_qx"], row["tgt_tcp_qy"], row["tgt_tcp_qz"] = tgt_tcp_quat
 
        pb_tcp_pos = data.get("pb_tcp_pos")
        if pb_tcp_pos is not None:
            row["pb_tcp_x"], row["pb_tcp_y"], row["pb_tcp_z"] = pb_tcp_pos
        pb_tcp_quat = data.get("pb_tcp_quat")
        if pb_tcp_quat is not None:
            row["pb_tcp_qw"], row["pb_tcp_qx"], row["pb_tcp_qy"], row["pb_tcp_qz"] = pb_tcp_quat
 
        q_tgt = data.get("q_tgt")
        if q_tgt is not None and len(q_tgt) >= 6:
            for i in range(6):
                row[f"q_tgt_{i+1}"] = q_tgt[i]
        pb_q = data.get("pb_q")
        if pb_q is not None and len(pb_q) >= 6:
            for i in range(6):
                row[f"pb_q_{i+1}"] = pb_q[i]
 
        for key in ("t_read_pose_ms", "t_pb_query_ms", "t_log_enqueue_ms",
                    "t_sleep_ms", "t_loop_total_ms"):
            if key in data:
                row[key] = data[key]
 
        self._queue.put_nowait(("control", row))
 
    # ------------------------------------------------------------------
    def close(self, timeout=5.0):
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._QUEUE_SENTINEL)
        self._worker_thread.join(timeout=timeout)
        if self._worker_thread.is_alive():
            print("[LOGGER WARN] Background writer thread did not exit in time; some log data may be lost")
        try:
            self._vision_file.flush()
            self._vision_file.close()
        except Exception:
            pass
        try:
            self._control_file.flush()
            self._control_file.close()
        except Exception:
            pass
 