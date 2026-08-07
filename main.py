import threading
import time
from datetime import datetime
import cv2

from src.shared_state import SharedPoseState, SharedJointState, SharedFrameState, SharedBoxPoseState, SharedPbState
from src.worker_perception import PerceptionWorker
from src.worker_curobo import cuRoboControllerWorker
from src.worker_pybullet import PyBulletSimulatorWorker
from src.module_logger import DataLogger

from tools.analyze_filter import analyze_vision
from tools.plot_vision import generate_report_figures
from tools.analyze_control import analyze_control
from tools.plot_control import generate_control_figures
from tools.analyze_frequency import analyze_frequency


class TeleopSystem:
    def __init__(self, playback_file=None):
        self.shared_pose = SharedPoseState()
        self.shared_joints = SharedJointState()
        self.shared_frame = SharedFrameState()
        self.shared_pybullet_frame = SharedFrameState()
        self.shared_box_poses = SharedBoxPoseState()
        self.shared_pb_state = SharedPbState()
        self.scene_ready = threading.Event()

        self.shared_log = None

        self.is_running = False
        self.threads = []

        self.playback_file = playback_file
        self.logger_vision_path = None
        self.logger_control_path = None

        self.robot_id = None
        self.arm_indices = []
        self.tcp_link_idx = 8

        self.system_ready = False
        self.teleop_active = False
        self.request_toggle_teleop = False
        self.last_space_press_time = 0.0

    def set_teleop_state(self, active_target):
        if self.system_ready and self.teleop_active != active_target:
            self.request_toggle_teleop = True

    def start_all(self):
        self.is_running = True

        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_prefix = "PB" if self.playback_file else "RT"
        log_prefix = f"{mode_prefix}_{current_time_str}"
        self.shared_log = DataLogger(log_prefix, out_dir="logs")
        self.logger_vision_path = self.shared_log.vision_path
        self.logger_control_path = self.shared_log.control_path

        self.perception_worker = PerceptionWorker(self)
        self.controller_worker = cuRoboControllerWorker(self)
        self.simulator_worker = PyBulletSimulatorWorker(self)

        self.threads = [
            threading.Thread(target=self.perception_worker.run, name="PerceptionTh", daemon=True),
            threading.Thread(target=self.controller_worker.run, name="ControllerTh", daemon=True),
            threading.Thread(target=self.simulator_worker.run, name="EGMTh", daemon=True),
        ]
        for t in self.threads:
            t.start()

    def stop_all(self):
        print("[System] Closing ...")
        self.is_running = False
        for t in self.threads:
            t.join()
        print("[System] All threads stopped.")

        if self.shared_log is not None:
            self.shared_log.close()

        if self.logger_vision_path and self.logger_control_path:
            print("[System] Analyzing log data...")
            analyze_vision(self.logger_vision_path)
            generate_report_figures(self.logger_vision_path, out_dir="figs")
            analyze_control(self.logger_control_path)
            generate_control_figures(self.logger_control_path, out_dir="figs")
            analyze_frequency(self.logger_control_path)

def main():
    system = TeleopSystem(playback_file=None)
    system.start_all()

    win_name = "Teleoperation Pipeline"
    robot_vision_win = "Robot Vision (Top: Wrist | Bottom: World)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(robot_vision_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(robot_vision_win, 320, 480)
    btn_w, btn_h = 300, 50
    space_time = 0.3
    last_pybullet_frame = None

    try:
        while True:
            display_frame = system.shared_frame.read()
            pybullet_frame = system.shared_pybullet_frame.read()
            if pybullet_frame is not None:
                last_pybullet_frame = pybullet_frame
            t_current = time.time()

            if system.teleop_active:
                if t_current - system.last_space_press_time > space_time:
                    system.set_teleop_state(active_target=False)

            if display_frame is not None:
                h, w, _ = display_frame.shape

                x2, y2 = w - 20, h - 20
                x1, y1 = x2 - btn_w, y2 - btn_h

                if not system.system_ready:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (100, 100, 100), cv2.FILLED)
                    cv2.putText(display_frame, "INITIALIZING...", (x1 + 15, y1 + 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    if not system.teleop_active:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 0), cv2.FILLED)
                        cv2.putText(display_frame, "HOLD 'SPACE' TO START", (x1 + 20, y1 + 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 220), cv2.FILLED)
                        cv2.putText(display_frame, "RELEASE 'SPACE' TO STOP", (x1 + 30, y1 + 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(win_name, display_frame)

            if last_pybullet_frame is not None:
                cv2.imshow(robot_vision_win, last_pybullet_frame)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                system.last_space_press_time = time.time()
                system.set_teleop_state(active_target=True)

    except KeyboardInterrupt:
        print("[SYSTEM] KeyboardInterrupt")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        system.stop_all()


if __name__ == "__main__":
    main()