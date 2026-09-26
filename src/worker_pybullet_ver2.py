"""
VER 2 — Debug window PyBullet được đặt TRÙNG góc nhìn world camera (quy đổi từ
cam_pos=[1.5, 0.0, 0.8], cam_target=[0.0, 0.0, 0.4] sang yaw/pitch/distance).
World camera KHÔNG render riêng nữa mà TÁI SỬ DỤNG chính view/proj matrix của
debug window đang hiển thị -> tiết kiệm 1 lần getCameraImage() mỗi tick.
Wrist camera hiện qua RGB buffer preview nhúng sẵn trong GUI của PyBullet
(COV_ENABLE_RGB_BUFFER_PREVIEW) -> không cần cửa sổ cv2 riêng.

  LƯU Ý QUAN TRỌNG:
  PyBullet KHÔNG có cách khoá người vận hành xoay/zoom camera bằng chuột.
  Vì world camera ở bản này lấy trực tiếp view matrix từ debug window đang hiển
  thị, nếu ai đó vô tình kéo chuột đổi góc nhìn giữa lúc record, world
  observation trong dataset sẽ TRÔI theo đúng thao tác đó, phá vỡ tính nhất
  quán viewpoint giữa các episode. Cần kỷ luật: không chạm chuột vào cửa sổ
  PyBullet trong lúc record thật. Nếu không chắc kiểm soát được điều này,
  dùng ver1 (world camera cố định, độc lập) sẽ an toàn hơn về chất lượng dữ liệu.
"""

import math
import time
import numpy as np
from pathlib import Path
import pybullet as p
import pybullet_data

from src.module_scene import SceneManager


def compute_gripper_control(current_pos, gripper_target, contact_detected=False,
                           open_limit=0.0, close_limit=0.71,
                           step=0.02, close_force=32.0, open_force=30.0, hold_force=12.0,
                           contact_step=0.001, contact_steps_after_touch=3, contact_phase=0):
    if gripper_target <= 0.5:
        target = max(current_pos - step, open_limit)
        force = open_force
        holding = False
    else:
        if contact_detected:
            if contact_phase < contact_steps_after_touch:
                target = min(current_pos + contact_step, close_limit)
                force = hold_force
                holding = True
            else:
                target = current_pos
                force = hold_force
                holding = True
        else:
            target = min(current_pos + step, close_limit)
            force = close_force
            holding = False

    return float(target), float(force), holding


def _eye_target_to_yaw_pitch_dist(eye, target):
    """Quy đổi (vị trí camera, điểm nhìn) -> (yaw, pitch, distance) cho resetDebugVisualizerCamera."""
    diff = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]
    distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
    yaw = math.degrees(math.atan2(diff[1], diff[0]))
    pitch = -math.degrees(math.asin(diff[2] / distance))  # camera ở trên nhìn xuống -> pitch âm
    return yaw, pitch, distance


class PyBulletSimulatorWorker:
    def __init__(self, system):
        self.sys = system
        self.axis_ids = [-1, -1, -1]
        self.scene_manager = SceneManager(table_top_z=0.4)
        self.gripper_contact_phase = 0

        self.wrist_rgb = None
        self.world_rgb = None

        # rs intrinsics
        self.cam_width = 640
        self.cam_height = 480
        self.cam_fx = 616.1630249023438
        self.cam_fy = 616.451416015625
        self.cam_cx = 314.67425537109375
        self.cam_cy = 247.34315490722656

        # world camera "gốc" — dùng để đặt góc debug window lúc khởi tạo.
        # Sau đó world camera thực tế = view matrix ĐANG HIỂN THỊ của debug window.
        self.WORLD_CAM_EYE = [1.5, 0.0, 0.8]
        self.WORLD_CAM_TARGET = [0.0, 0.0, 0.4]

        # record
        self.RECORD_HZ = 15.0
        self.RECORD_PERIOD = 1.0 / self.RECORD_HZ
        self._next_record_deadline = None
        self._last_record_tick_t = None

    def _compute_projection_matrix(self, near, far):
        width = self.cam_width
        height = self.cam_height

        projection_mat = [
            2.0 * self.cam_fx / width, 0.0, 0.0, 0.0,
            0.0, 2.0 * self.cam_fy / height, 0.0, 0.0,
            1.0 - 2.0 * self.cam_cx / width, (2.0 * self.cam_cy / height) - 1.0,
            (far + near) / (near - far), -1.0,
            0.0, 0.0, (2.0 * far * near) / (near - far), 0.0
        ]
        return tuple(projection_mat)

    def _get_wrist_camera_data(self, link_state):
        link_pos = link_state[4]
        link_ori = link_state[5]

        relative_translation = [0.0, 0.025, 0.05]
        relative_rotation = p.getQuaternionFromEuler([0, 0, 0])
        cam_pos, cam_ori = p.multiplyTransforms(link_pos, link_ori, relative_translation, relative_rotation)

        rot_matrix = np.reshape(p.getMatrixFromQuaternion(cam_ori), (3, 3))
        forward_vec = rot_matrix[:, 2]
        up_vec = rot_matrix[:, 1]

        cam_target = np.array(cam_pos) + forward_vec * 0.5
        view_mat = p.computeViewMatrix(cam_pos, cam_target.tolist(), up_vec.tolist())
        proj_mat = self._compute_projection_matrix(near=0.01, far=1.0)

        # Đây là lần getCameraImage() CUỐI CÙNG trong mỗi tick -> RGB buffer preview
        # nhúng trong GUI sẽ hiển thị đúng ảnh wrist (preview luôn show buffer mới nhất).
        images = p.getCameraImage(
            self.cam_width, self.cam_height,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK,
            shadow=0
        )
        rgba = np.reshape(np.array(images[2], dtype=np.uint8), (self.cam_height, self.cam_width, -1))
        return rgba[:, :, 0:3]

    def _get_world_camera_data_from_debug_view(self):
        """Tái sử dụng view/proj matrix của chính debug window đang hiển thị,
        thay vì tính view matrix cố định riêng -> tiết kiệm 1 lần render độc lập."""
        debug_cam = p.getDebugVisualizerCamera()
        debug_view_mat, debug_proj_mat = debug_cam[2], debug_cam[3]

        images = p.getCameraImage(
            self.cam_width, self.cam_height,
            viewMatrix=debug_view_mat,
            projectionMatrix=debug_proj_mat,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK,
            shadow=0
        )
        rgba = np.reshape(np.array(images[2], dtype=np.uint8), (self.cam_height, self.cam_width, -1))
        return rgba[:, :, 0:3]

    def _draw_tcp_axes(self, link_state, axis_length=0.15):
        tcp_pos = link_state[4]
        tcp_orn_xyzw = link_state[5]

        rot_matrix = np.reshape(p.getMatrixFromQuaternion(tcp_orn_xyzw), (3, 3))

        x_end = np.array(tcp_pos) + rot_matrix[:, 0] * axis_length
        y_end = np.array(tcp_pos) + rot_matrix[:, 1] * axis_length
        z_end = np.array(tcp_pos) + rot_matrix[:, 2] * axis_length

        if self.axis_ids[0] < 0:
            self.axis_ids[0] = p.addUserDebugLine(tcp_pos, x_end.tolist(), [1, 0, 0], 3)
            self.axis_ids[1] = p.addUserDebugLine(tcp_pos, y_end.tolist(), [0, 1, 0], 3)
            self.axis_ids[2] = p.addUserDebugLine(tcp_pos, z_end.tolist(), [0, 0, 1], 3)
        else:
            p.addUserDebugLine(tcp_pos, x_end.tolist(), [1, 0, 0], 3, replaceItemUniqueId=self.axis_ids[0])
            p.addUserDebugLine(tcp_pos, y_end.tolist(), [0, 1, 0], 3, replaceItemUniqueId=self.axis_ids[1])
            p.addUserDebugLine(tcp_pos, z_end.tolist(), [0, 0, 1], 3, replaceItemUniqueId=self.axis_ids[2])

    def _has_box_contact(self):
        if self.sys.robot_id is None:
            return False
        try:
            for box_id in self.scene_manager.box_ids:
                contacts = p.getContactPoints(self.sys.robot_id, box_id)
                for contact in contacts:
                    if contact[8] <= 0.002:
                        return True
        except (p.error, AttributeError):
            return False
        return False

    def run(self):
        print("[Communication] Initializing...")
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.scene_manager.setup_pick_and_place_scene()
        box_poses = []
        for i, box_id in enumerate(self.scene_manager.box_ids):
            pos, orn_xyzw = p.getBasePositionAndOrientation(box_id)
            box_poses.append({
                "name": f"box_{i}",
                "position": list(pos),
                "quaternion_xyzw": list(orn_xyzw),
            })
        self.sys.shared_box_poses.write(box_poses)
        self.sys.scene_ready.set()
        print(f"[Communication] Synced {len(box_poses)} box poses to shared state")

        # VER2: đặt góc debug window trùng world camera + bật preview wrist nhúng sẵn
        yaw, pitch, distance = _eye_target_to_yaw_pitch_dist(self.WORLD_CAM_EYE, self.WORLD_CAM_TARGET)
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=self.WORLD_CAM_TARGET,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        urdf_path = str(Path("assets/abb_irb1200_509_gripper/irb1200_full.urdf").resolve())
        self.sys.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

        name_to_index = {}
        for i in range(p.getNumJoints(self.sys.robot_id)):
            info = p.getJointInfo(self.sys.robot_id, i)
            name_to_index[info[1].decode("utf-8")] = i
        self.sys.tcp_link_idx = name_to_index.get("robotiq_tcp_joint", 8)
        self.sys.tool0_link_idx = name_to_index.get("tool0", 7)
        print(f"[Communication] TCP - Link ID={self.sys.tcp_link_idx} ({[k for k,v in name_to_index.items() if v==self.sys.tcp_link_idx]})")

        gripper_signed_limits = {
            "finger_joint":              (0.0, 0.725),
            "left_inner_knuckle_joint":  (0.0, 0.8757),
            "right_inner_finger_joint":  (0.0, 0.8757),
            "left_inner_finger_joint":   (-0.8757, 0.0),
            "right_outer_knuckle_joint": (-0.8757, 0.0),
            "right_inner_knuckle_joint": (-0.8757, 0.0),
        }
        for name, (lo, hi) in gripper_signed_limits.items():
            if name in name_to_index:
                p.changeDynamics(self.sys.robot_id, name_to_index[name], jointLowerLimit=lo, jointUpperLimit=hi)

        for i in range(p.getNumJoints(self.sys.robot_id)):
            p.changeDynamics(self.sys.robot_id, i, lateralFriction=0.6, spinningFriction=0.2,
                             rollingFriction=0.0001, frictionAnchor=True)

        arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        JOINT_MAX_VEL = {
                    "joint_1": 5.03,   # 288 deg/s
                    "joint_2": 4.19,   # 240
                    "joint_3": 5.18,   # 297
                    "joint_4": 6.56,   # 376
                    "joint_5": 6.96,   # 399
                    "joint_6": 10.47,  # 600
                }
        arm_max_vel = [JOINT_MAX_VEL[name] for name in arm_joint_names if name in name_to_index]
        self.sys.arm_indices = [name_to_index[name] for name in arm_joint_names if name in name_to_index]

        master_idx = name_to_index["finger_joint"]

        slave_joint_names = ["left_inner_knuckle_joint", "left_inner_finger_joint",
                             "right_outer_knuckle_joint", "right_inner_knuckle_joint",
                             "right_inner_finger_joint"]
        slave_indices = [name_to_index[name] for name in slave_joint_names if name in name_to_index]
        mimic_multipliers = np.array([1.0, -1.0, -1.0, -1.0, 1.0])

        home_joints_deg = [0.0, 0.0, 0.0, 0.0, 90.0, 90.0]
        home_joints_rad = [math.radians(deg) for deg in home_joints_deg]

        for idx, target_rad in zip(self.sys.arm_indices, home_joints_rad):
            p.resetJointState(self.sys.robot_id, idx, target_rad)

        self.sys.shared_joints.write(home_joints_rad + [0.0], time.time())

        print("[Communication] PyBullet simulation started")
        log_counter = 0

        try:
            for idx in slave_indices + [master_idx]:
                try:
                    p.changeDynamics(self.sys.robot_id, idx, lateralFriction=1.6, spinningFriction=0.005,
                                        rollingFriction=0.0005, restitution=0.0, contactProcessingThreshold=0.001)
                except Exception:
                    print(f"[Communication] Failed to change dynamics for joint index {idx}")
                    pass
        except Exception:
            print("[Communication] Failed to change dynamics for one or more joints")
            pass

        try:
            while self.sys.is_running:
                joints = None

                if self.sys.robot_id is not None:
                    try:
                        tcp_link_state = p.getLinkState(self.sys.robot_id, self.sys.tcp_link_idx, computeForwardKinematics=True)
                        pb_state = {
                            "tcp_pos": list(tcp_link_state[4]),
                            "tcp_quat_wxyz": [tcp_link_state[5][3], tcp_link_state[5][0],
                                                tcp_link_state[5][1], tcp_link_state[5][2]],
                            "q": [p.getJointState(self.sys.robot_id, idx)[0] for idx in self.sys.arm_indices],
                        }
                        self.sys.shared_pb_state.write(pb_state, time.time())
                        joints, _ts = self.sys.shared_joints.read()

                        now = time.perf_counter()
                        if self._next_record_deadline is None:
                            self._next_record_deadline = now

                        if now >= self._next_record_deadline:
                            t_tick_start = time.perf_counter()
                            actual_hz = (1.0 / (now - self._last_record_tick_t)
                                        if self._last_record_tick_t is not None else None)
                            self._last_record_tick_t = now

                            tool0_link_state = p.getLinkState(self.sys.robot_id, self.sys.tool0_link_idx, computeForwardKinematics=True)

                            # --- world TRƯỚC (tái dùng view matrix debug window) ---
                            t_c1 = time.perf_counter()
                            self.world_rgb = self._get_world_camera_data_from_debug_view()
                            t_world_ms = (time.perf_counter() - t_c1) * 1000.0

                            # --- wrist SAU CÙNG -> RGB buffer preview trong GUI hiện đúng ảnh wrist ---
                            t_c0 = time.perf_counter()
                            self.wrist_rgb = self._get_wrist_camera_data(tool0_link_state)
                            t_wrist_ms = (time.perf_counter() - t_c0) * 1000.0

                            t_tick_total_ms = (time.perf_counter() - t_tick_start) * 1000.0

                            if hasattr(self.sys, "shared_log") and self.sys.shared_log is not None:
                                self.sys.shared_log.write_record({
                                    "actual_hz": actual_hz,
                                    "t_wrist_capture_ms": t_wrist_ms,
                                    "t_world_capture_ms": t_world_ms,
                                    "t_tick_total_ms": t_tick_total_ms,
                                    "action": joints,
                                    "pb_state": pb_state,
                                    # dataset.add_frame({..., "observation.images.wrist": self.wrist_rgb,
                                    #                    "observation.images.world": self.world_rgb, ...})
                                })

                            self._next_record_deadline += self.RECORD_PERIOD
                            if self._next_record_deadline < now:
                                self._next_record_deadline = now + self.RECORD_PERIOD

                    except p.error:
                        pass

                if joints is not None and len(joints) >= 7:
                    arm_targets = joints[:6]
                    gripper_target = joints[6]

                    for idx, target, max_vel in zip(self.sys.arm_indices, arm_targets, arm_max_vel):
                        p.setJointMotorControl2(self.sys.robot_id, idx, p.POSITION_CONTROL, targetPosition=target, force=200,
                                                maxVelocity=max_vel)

                    actual_master_pos = p.getJointState(self.sys.robot_id, master_idx)[0]

                    contact_detected = self._has_box_contact()
                    if gripper_target > 0.5 and contact_detected:
                        self.gripper_contact_phase = min(self.gripper_contact_phase + 1, 5)
                    else:
                        self.gripper_contact_phase = 0

                    target_master_pos, current_force, _ = compute_gripper_control(
                        current_pos=actual_master_pos,
                        gripper_target=gripper_target,
                        contact_detected=contact_detected,
                        contact_phase=self.gripper_contact_phase,
                    )

                    p.setJointMotorControl2(
                        self.sys.robot_id,
                        master_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target_master_pos,
                        force=current_force,
                        maxVelocity=4.0,
                    )

                    target_slave_positions = (mimic_multipliers * target_master_pos).tolist()
                    p.setJointMotorControlArray(
                        self.sys.robot_id,
                        slave_indices,
                        p.POSITION_CONTROL,
                        targetPositions=target_slave_positions,
                        forces=[current_force] * len(slave_indices),
                        positionGains=[0.8] * len(slave_indices),
                    )

                    log_counter += 1
                    if self.sys.teleop_active and log_counter % 12 == 0:
                        current_pybullet_angles = [p.getJointState(self.sys.robot_id, idx)[0] for idx in self.sys.arm_indices]
                        print(f"[PyBullet-EXEC] Target: {[round(t, 2) for t in arm_targets]} | Actual: {[round(a, 2) for a in current_pybullet_angles]}")

                loop_start = time.perf_counter()
                p.stepSimulation()
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, (1.0/240.0) - elapsed)
                time.sleep(sleep_time)

        finally:
            print("[Communication] Closing PyBullet...")


# ---------------------------------------------------------------------------
# GHI CHÚ:
# 1. Trong main.py, xoá đoạn tạo "Robot Vision" window + đọc shared_pybullet_frame
#    (giống ver1) — cửa sổ debug PyBullet giờ tự đóng vai trò world view + wrist preview.
# 2. Nếu preview wrist không hiện (tuỳ driver/GPU), thử bật lại
#    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1) để xem toolbar + preview panel đầy đủ,
#    rồi tắt lại COV_ENABLE_GUI sau khi xác nhận preview hoạt động.
# 3. KHÔNG chạm chuột vào cửa sổ PyBullet trong lúc record thật (xem cảnh báo đầu file).
# ---------------------------------------------------------------------------
