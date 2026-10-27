import threading
from datetime import datetime
import cv2
import ctypes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import traceback

from src.shared_state import SharedPoseState, SharedJointState, SharedFrameState, SharedBoxPoseState, SharedPbState
from src.worker_perception import PerceptionWorker
from src.worker_curobo import cuRoboControllerWorker
from src.worker_pybullet import PyBulletSimulatorWorker
from src.module_logger import DataLogger

from tools.analyze_filter import analyze_vision
from tools.plot_vision import generate_report_figures
from tools.analyze_control import analyze_control
from tools.plot_control import generate_control_figures
from tools.analyze_frequency import analyze_frequency, analyze_record_frequency,analyze_vision_frequency, correlate_long_frames

# call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 

user32 = ctypes.windll.user32
VK_SPACE = 0x20
JOINT_KEYS = {ord('4'): 4, ord('5'): 5, ord('6'): 6}

def _key_down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)

def _safe(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception:
        print(f"[System] {fn.__name__} failed:")
        traceback.print_exc()

class TeleopSystem:
    def __init__(self, playback_file):
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
        self.logger_record_path = None

        self.robot_id = None
        self.arm_indices = []
        self.tcp_link_idx = 8

        self.system_ready = False
        self.teleop_active = False
        self.request_toggle_teleop = False
        self.selected_joint = 6

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
        self.logger_record_path = self.shared_log.record_path

        self.perception_worker = PerceptionWorker(self)
        self.controller_worker = cuRoboControllerWorker(self)
        self.simulator_worker = PyBulletSimulatorWorker(self)

        dataset_features = {
            "action": {"dtype": "float32", "shape": (7,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
            "observation.images.wrist": {"dtype": "video", "shape": (72, 96, 3)},
            "observation.images.world": {"dtype": "video", "shape": (192, 256, 3)},
        }

        # self.dataset = LeRobotDataset.create(
        #     repo_id="bthw/lerobot-dataset", 
        #     fps=9,
        #     features=dataset_features,
        #     robot_type="abb_irb1200_pybullet",
        #     use_videos=True,
        #     image_writer_threads=4,
        #     tolerance_s=0.1,
        #     root='./dataset',
        # )

        self.dataset = LeRobotDataset.resume(
            repo_id="bthw/lerobot-dataset",
            root='./dataset',
            tolerance_s=0.1,
            image_writer_threads=4,
        )

        self.task_str = "pick up the box and place it in the target zone"
        self.request_save_episode = False
        self.request_clear_episode = False

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

        if hasattr(self, "dataset") and self.dataset is not None:
            self.dataset.finalize()

        print("[System] Analyzing log data...")
        v, c, r = self.logger_vision_path, self.logger_control_path, self.logger_record_path

        if v:
            _safe(analyze_vision, v)
            _safe(generate_report_figures, v, out_dir="figs")
            _safe(analyze_vision_frequency, v)
        if c:
            _safe(analyze_control, c)
            _safe(generate_control_figures, c, out_dir="figs")
            _safe(analyze_frequency, c)
        if r:
            _safe(analyze_record_frequency, r, target_hz=10.0)
        if v and r:
            _safe(correlate_long_frames, v, r)

def main():
    system = TeleopSystem(playback_file=None)  
    system.start_all()

    win_name = "Teleoperation Pipeline"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    robot_vision_win = "Robot Vision (Wrist)"
    window_open = False
    btn_w, btn_h = 300, 50
    prev_down = {vk: False for vk in JOINT_KEYS}
    last_pybullet_frame = None

    try:
        while True:
            display_frame = system.shared_frame.read()

            hwnd = user32.FindWindowW(None, win_name)
            focused = hwnd != 0 and user32.GetForegroundWindow() == hwnd
            system.set_teleop_state(active_target=focused and _key_down(VK_SPACE))
            for vk, joint in JOINT_KEYS.items():
                down = focused and _key_down(vk)
                if down and not prev_down[vk]:
                    system.selected_joint = joint
                prev_down[vk] = down

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

                cv2.putText(display_frame, f"JOINT: J{system.selected_joint}  (keys 4/5/6)", (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)

                cv2.imshow(win_name, display_frame)

            pybullet_frame = system.shared_pybullet_frame.read()
            if pybullet_frame is not None:
                last_pybullet_frame = pybullet_frame
                if not window_open:
                    cv2.namedWindow(robot_vision_win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(robot_vision_win, 96, 72)
                    window_open = True
                cv2.imshow(robot_vision_win, last_pybullet_frame)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('n'):
                system.request_save_episode = True
            elif key == ord('r'):
                system.request_clear_episode = True

    except KeyboardInterrupt:
        print("[SYSTEM] KeyboardInterrupt")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        system.stop_all()


if __name__ == "__main__":
    winmm = ctypes.windll.winmm
    winmm.timeBeginPeriod(1)
    try:
        main()
    finally:
        winmm.timeEndPeriod(1)