import time
import cv2
import pyrealsense2 as rs

from src.module_camera import RealSenseNode
from src.module_tracker import HandTrackerNode
from src.module_filter import Position3DFilter, GestureDebouncer

GESTURE_ROT = {"Pointing_Up": 1, "Victory": -1}                  # CW = +1, CCW = -1
GESTURE_GRIPPER = {"Open_Palm": "Open", "Closed_Fist": "Close"}
ROT_TEXT = {1: "CW", -1: "CCW", 0: "-"}

class PerceptionWorker:
    def __init__(self, system):
        self.sys = system

    def run(self):
        print("[Perception] Initializing...")
        
        TCP_INDEX = 0
        gripper_state = "Open"
        
        camera = RealSenseNode(playback_file=self.sys.playback_file)
        tracker = HandTrackerNode(model_path='model/gesture_recognizer.task')
        
        tcp_filter = Position3DFilter(min_cutoff=0.005, beta=20.0, cutoff_max=15.0, reject_max_jump_mps=2.18, slew_limit_mps=1.0)
        debouncer = GestureDebouncer(min_score=0.7, press_s=0.15, release_s=0.06)
        
        try:
            while self.sys.is_running:
                color_img, depth_frame, depth_arr, timestamp = camera.get_frames()
                if color_img is None:
                    continue
                t_s = timestamp / 1000.0

                tracker.recognize_async(color_img, timestamp)
                color_img = tracker.draw_skeleton(color_img)
                landmarks = tracker.get_all_landmarks_pixel(color_img)
                gesture_raw, gesture_score = tracker.get_gesture()
                gesture_valid = gesture_raw if (gesture_raw in GESTURE_ROT or gesture_raw in GESTURE_GRIPPER) else None

                rot_cmd = 0
                confirmed = None
                if landmarks:
                    if self.sys.teleop_active:
                        confirmed = debouncer.update(gesture_valid, gesture_score, t_s)
                    else:
                        debouncer.reset()
                    rot_cmd = GESTURE_ROT.get(confirmed, 0)
                    if confirmed in GESTURE_GRIPPER:
                        gripper_state = GESTURE_GRIPPER[confirmed]

                log_data_dict = {
                    "frame_timestamp_s": t_s,
                    "raw_pos": None, "filt_pos": None,
                    "gesture_raw": gesture_raw, "gesture_score": gesture_score,
                    "rot_cmd": rot_cmd, "gripper_state": gripper_state,
                }

                if landmarks:
                    g_color = (0, 0, 255) if gripper_state == "Close" else (0, 255, 0)
                    cv2.putText(color_img, f"Gripper: {gripper_state}", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, g_color, 2, cv2.LINE_AA)
                    cv2.putText(color_img, f"Gesture: {gesture_raw or '-'} ({gesture_score:.2f}) -> {confirmed or '-'}", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(color_img, f"Rot: {ROT_TEXT[rot_cmd]}", (20, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

                    uTCP, vTCP = landmarks[TCP_INDEX]
                    P_TCP_3D = camera.extract_3d_coordinates(uTCP, vTCP, depth_frame, depth_arr)

                    if P_TCP_3D:
                        log_data_dict["raw_pos"] = P_TCP_3D
                        P_TCP_3D = tcp_filter.filter(P_TCP_3D, t_s)
                        log_data_dict["filt_pos"] = P_TCP_3D
                        
                        u_disp, v_disp = rs.rs2_project_point_to_pixel(camera.intrinsics, P_TCP_3D)
                        cv2.circle(color_img, (uTCP, vTCP), 8, (0, 255, 255), cv2.FILLED)
                        cv2.circle(color_img, (int(u_disp), int(v_disp)), 8, (255, 0, 255), cv2.FILLED)
                        cv2.putText(color_img, f"TCP: X:{P_TCP_3D[0]:.3f} Y:{P_TCP_3D[1]:.3f} Z:{P_TCP_3D[2]:.3f} m", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                        pose_data = {"position": P_TCP_3D, "rot_cmd": rot_cmd, "gripper": gripper_state}
                        self.sys.shared_pose.write(pose_data, t_s)
                    else:
                        cv2.putText(color_img, "[Perception WARN]: NO TCP DEPTH", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(color_img, "[Perception WARN]: NO HAND DETECTED", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    tcp_filter.reset()
                    debouncer.reset()

                self.sys.shared_log.write_vision(log_data_dict)

                self.sys.shared_frame.write(color_img)
                time.sleep(0.01)

        finally:
            print("[Perception] Closing ...")
            tracker.close()
            camera.stop()