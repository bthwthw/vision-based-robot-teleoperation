import csv
import os
import time
import threading
import queue
 
 
class DataLogger:
    VISION_FIELDNAMES = [
        "wall_time_s", "frame_timestamp_s",
        "raw_x", "raw_y", "raw_z", "filt_x", "filt_y", "filt_z",
        "gesture_raw", "gesture_score", "rot_cmd", "gripper_state", "teleop_active",
    ]
 
    CONTROL_FIELDNAMES = [
        "wall_time_s", "frame_timestamp_s",
        "curobo_time_ms", "ik_success",
        "selected_joint", "rot_cmd", "rot_committed",
        "tgt_tcp_x", "tgt_tcp_y", "tgt_tcp_z",
        "tgt_tcp_qw", "tgt_tcp_qx", "tgt_tcp_qy", "tgt_tcp_qz",
        "pb_tcp_x", "pb_tcp_y", "pb_tcp_z",
        "pb_tcp_qw", "pb_tcp_qx", "pb_tcp_qy", "pb_tcp_qz",
        "q_tgt_1", "q_tgt_2", "q_tgt_3", "q_tgt_4", "q_tgt_5", "q_tgt_6",
        "pb_q_1", "pb_q_2", "pb_q_3", "pb_q_4", "pb_q_5", "pb_q_6",
        "t_read_pose_ms", "t_pb_query_ms", "t_log_enqueue_ms",
        "t_sleep_ms", "t_loop_total_ms",
    ]

    RECORD_FIELDNAMES = [
        "wall_time_s", "actual_hz",
        "t_wrist_capture_ms", "t_world_capture_ms", "t_tick_total_ms",
        "action_1", "action_2", "action_3", "action_4", "action_5", "action_6", "action_gripper",
        "state_1", "state_2", "state_3", "state_4", "state_5", "state_6",
        "state_tcp_x", "state_tcp_y", "state_tcp_z",
        "state_tcp_qw", "state_tcp_qx", "state_tcp_qy", "state_tcp_qz",
    ]
 
    _QUEUE_SENTINEL = object()
 
    def __init__(self, filename_prefix, out_dir="logs", flush_every_n=50):
        os.makedirs(out_dir, exist_ok=True)
        self._start_wall_time = time.time()
 
        if filename_prefix.endswith(".csv"):
            filename_prefix = filename_prefix[:-4]
 
        self.vision_path = os.path.join(out_dir, f"{filename_prefix}_vision.csv")
        self._vision_file = open(self.vision_path, mode="w", newline="", buffering=1 << 16)
        self._vision_writer = csv.DictWriter(self._vision_file, fieldnames=self.VISION_FIELDNAMES)
        self._vision_writer.writeheader()
        self._vision_count = 0

        self.control_path = os.path.join(out_dir, f"{filename_prefix}_control.csv")
        self._control_file = open(self.control_path, mode="w", newline="", buffering=1 << 16)
        self._control_writer = csv.DictWriter(self._control_file, fieldnames=self.CONTROL_FIELDNAMES)
        self._control_writer.writeheader()
        self._control_count = 0

        self.record_path = os.path.join(out_dir, f"{filename_prefix}_record.csv")
        self._record_file = open(self.record_path, mode="w", newline="", buffering=1 << 16)
        self._record_writer = csv.DictWriter(self._record_file, fieldnames=self.RECORD_FIELDNAMES)
        self._record_writer.writeheader()
        self._record_count = 0
 
        self._flush_every_n = flush_every_n
 
        self._queue = queue.Queue()
        self._closed = False
        self._worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker_thread.start()
 
        print(f"[LOGGER INFO] Vision  log -> {self.vision_path}")
        print(f"[LOGGER INFO] Control log -> {self.control_path}")
        print(f"[LOGGER INFO] Record log -> {self.record_path}")

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
                elif kind == "record":
                    self._record_writer.writerow(row)
                    self._record_count += 1
                    if self._record_count % self._flush_every_n == 0:
                        self._record_file.flush()
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
 
        row["gesture_raw"] = data.get("gesture_raw")
        row["gesture_score"] = data.get("gesture_score")
        row["rot_cmd"] = data.get("rot_cmd")
 
        gripper_state = data.get("gripper_state")
        row["gripper_state"] = None if gripper_state is None else int(gripper_state == "Close")
        teleop_active = data.get("teleop_active")
        row["teleop_active"] = None if teleop_active is None else int(teleop_active)
 
        self._queue.put_nowait(("vision", row))
 
    def write_control(self, data):
        row = {
            "wall_time_s": round(time.time() - self._start_wall_time, 6),
            "frame_timestamp_s": data.get("frame_timestamp_s"),
            "curobo_time_ms": data.get("curobo_time_ms"),
            "ik_success": data.get("ik_success"),
            "selected_joint": data.get("selected_joint"),
            "rot_cmd": data.get("rot_cmd"),
            "rot_committed": data.get("rot_committed"),
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

    def write_record(self, data):
        row = {
            "wall_time_s": round(time.time() - self._start_wall_time, 6),
            "actual_hz": data.get("actual_hz"),
            "t_wrist_capture_ms": data.get("t_wrist_capture_ms"),
            "t_world_capture_ms": data.get("t_world_capture_ms"),
            "t_tick_total_ms": data.get("t_tick_total_ms"),
        }
        action = data.get("action")
        if action is not None and len(action) >= 7:
            for i in range(6):
                row[f"action_{i+1}"] = action[i]
            row["action_gripper"] = action[6]

        pb_state = data.get("pb_state")
        if pb_state is not None:
            q = pb_state.get("q")
            if q is not None and len(q) >= 6:
                for i in range(6):
                    row[f"state_{i+1}"] = q[i]
            pos = pb_state.get("tcp_pos")
            if pos is not None:
                row["state_tcp_x"], row["state_tcp_y"], row["state_tcp_z"] = pos
            quat = pb_state.get("tcp_quat_wxyz")
            if quat is not None:
                row["state_tcp_qw"], row["state_tcp_qx"], row["state_tcp_qy"], row["state_tcp_qz"] = quat

        self._queue.put_nowait(("record", row))

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
        try:
            self._record_file.flush()
            self._record_file.close()
        except Exception:
            pass