import math
import time
import cv2
import numpy as np
from pathlib import Path
import pybullet as p
import pybullet_data

from src.module_scene import SceneManager

def _eye_target_to_yaw_pitch_dist(eye, target):
    diff = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]
    distance = math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
    yaw = math.degrees(math.atan2(diff[1], diff[0]))
    pitch = -math.degrees(math.asin(diff[2] / distance))
    return yaw, pitch, distance

def compute_gripper_control(current_pos, gripper_target, contact_detected=False,
                           open_limit=0.0, close_limit=0.71,
                           step=0.02, close_force=32.0, open_force=30.0, hold_force=25.0):
    if gripper_target <= 0.5:
        target = max(current_pos - step, open_limit)
        force = open_force
        holding = False
    else:
        if contact_detected:
            # Luôn target vị trí đóng kín tối đa — vật thể vật lý chặn lại,
            # sai số vị trí không bao giờ về 0 -> motor liên tục ép hold_force vào vật
            target = close_limit
            force = hold_force
            holding = True
        else:
            target = min(current_pos + step, close_limit)
            force = close_force
            holding = False

    return float(target), float(force), holding

class PyBulletSimulatorWorker:
    def __init__(self, system):
        self.sys = system
        self.axis_ids = [-1, -1, -1]
        self.scene_manager = SceneManager(table_top_z=0.4)
        self.gripper_contact_phase = 0
        # self.wrist_bgr = None
        # self.world_bgr = None

        # rs intrinsics 
        self.rs_base_width = 640
        self.rs_base_height = 480
        self.cam_fx = 616.1630249023438
        self.cam_fy = 616.451416015625
        self.cam_cx = 314.67425537109375
        self.cam_cy = 247.34315490722656

        # world (static) camera
        self.world_cam_width = 256
        self.world_cam_height = 192

        # wrist (gripper) camera 
        self.wrist_cam_width = 96
        self.wrist_cam_height = 72

        # record
        self.RECORD_HZ = 10.0
        self.RECORD_PERIOD = 1.0 / self.RECORD_HZ
        self._next_record_deadline = None
        self._last_record_tick_t = None

    def _compute_projection_matrix(self, width, height, near, far):
        scale_x = width / self.rs_base_width
        scale_y = height / self.rs_base_height
        fx = self.cam_fx * scale_x
        fy = self.cam_fy * scale_y
        cx = self.cam_cx * scale_x
        cy = self.cam_cy * scale_y

        projection_mat = [
            2.0 * fx / width, 0.0, 0.0, 0.0,
            0.0, 2.0 * fy / height, 0.0, 0.0,
            1.0 - 2.0 * cx / width, (2.0 * cy / height) - 1.0,
            (far + near) / (near - far), -1.0,
            0.0, 0.0, (2.0 * far * near) / (near - far), 0.0
            ]

        return tuple(projection_mat)
    
    def _get_wrist_camera_data(self, link_state):
        link_pos = link_state[4]
        link_ori = link_state[5]
        
        # Extrinsics tương đối so với tool0
        relative_translation = [0.0, 0.025, 0.05] 
        relative_rotation = p.getQuaternionFromEuler([0, 0, 0])
        cam_pos, cam_ori = p.multiplyTransforms(link_pos, link_ori, relative_translation, relative_rotation)
        # self._draw_camera_axes(cam_pos, cam_ori)
        # self._draw_camera_frustum(cam_pos, cam_ori, fov_deg=60, aspect=self.wrist_cam_width / self.wrist_cam_height, far=1.0, tag="wrist")

        rot_matrix = np.reshape(p.getMatrixFromQuaternion(cam_ori), (3, 3))
        
        # Camera coordinate system: forward = z_pybullet, up = y_pybullet
        forward_vec = rot_matrix[:, 2] 
        up_vec = rot_matrix[:, 1]  
        
        cam_target = np.array(cam_pos) + forward_vec * 0.5
        view_mat = p.computeViewMatrix(cam_pos, cam_target.tolist(), up_vec.tolist())
        
        proj_mat = self._compute_projection_matrix(self.wrist_cam_width, self.wrist_cam_height, near=0.01, far=1.0)
        
        images = p.getCameraImage(  self.wrist_cam_width, self.wrist_cam_height,
                                    viewMatrix=view_mat,
                                    projectionMatrix=proj_mat,
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                    flags=p.ER_NO_SEGMENTATION_MASK,
                                    shadow=0)

        rgba = np.reshape(np.array(images[2], dtype=np.uint8),
                        (self.wrist_cam_height, self.wrist_cam_width, -1))
        depth_buffer = np.reshape(np.array(images[3], dtype=np.float32), (self.wrist_cam_height, self.wrist_cam_width))
        # depth_meters = far * near / (far - (far - near) * depth_buffer)

        return rgba[:, :, 0:3]

    def _get_world_camera_data(self):
        camera = p.getDebugVisualizerCamera()
        view_matrix = camera[2]
        projection_matrix = camera[3]

        image = p.getCameraImage(
            self.world_cam_width,
            self.world_cam_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK,
            shadow=0,
        )

        rgba = np.asarray(image[2], dtype=np.uint8).reshape(
            self.world_cam_height, self.world_cam_width, 4
        )
        return rgba[:, :, :3]

    def _draw_camera_frustum(self, cam_pos, cam_ori, fov_deg, aspect, far, tag="wrist"):
        if not hasattr(self, '_frustum_ids'):
            self._frustum_ids = {}
        if tag not in self._frustum_ids:
            self._frustum_ids[tag] = [-1, -1, -1, -1]

        rot_matrix = np.reshape(p.getMatrixFromQuaternion(cam_ori), (3, 3))
        forward = rot_matrix[:, 2]
        right = rot_matrix[:, 0]
        up = rot_matrix[:, 1]

        half_h = far * math.tan(math.radians(fov_deg) / 2.0)
        half_w = half_h * aspect

        center_far = np.array(cam_pos) + forward * far
        corners = [
            center_far + up * half_h + right * half_w,
            center_far + up * half_h - right * half_w,
            center_far - up * half_h - right * half_w,
            center_far - up * half_h + right * half_w,
        ]

        for i, corner in enumerate(corners):
            if self._frustum_ids[tag][i] < 0:
                self._frustum_ids[tag][i] = p.addUserDebugLine(cam_pos, corner.tolist(), [1, 1, 0], 1)
            else:
                p.addUserDebugLine(cam_pos, corner.tolist(), [1, 1, 0], 1,
                                    replaceItemUniqueId=self._frustum_ids[tag][i])
            
    def _draw_camera_axes(self, cam_pos, cam_ori, axis_length=0.1):
        if not hasattr(self, 'cam_axis_ids'):
            self.cam_axis_ids = [-1, -1, -1]

        rot_matrix = np.reshape(p.getMatrixFromQuaternion(cam_ori), (3, 3))

        x_end = np.array(cam_pos) + rot_matrix[:, 0] * axis_length
        y_end = np.array(cam_pos) + rot_matrix[:, 1] * axis_length
        z_end = np.array(cam_pos) + rot_matrix[:, 2] * axis_length

        if self.cam_axis_ids[0] < 0:
            self.cam_axis_ids[0] = p.addUserDebugLine(cam_pos, x_end.tolist(), [1, 0, 0], 2)
            self.cam_axis_ids[1] = p.addUserDebugLine(cam_pos, y_end.tolist(), [0, 1, 0], 2)
            self.cam_axis_ids[2] = p.addUserDebugLine(cam_pos, z_end.tolist(), [0, 0, 1], 2)
        else:
            p.addUserDebugLine(cam_pos, x_end.tolist(), [1, 0, 0], 2, replaceItemUniqueId=self.cam_axis_ids[0])
            p.addUserDebugLine(cam_pos, y_end.tolist(), [0, 1, 0], 2, replaceItemUniqueId=self.cam_axis_ids[1])
            p.addUserDebugLine(cam_pos, z_end.tolist(), [0, 0, 1], 2, replaceItemUniqueId=self.cam_axis_ids[2])
    
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
            return False, False

        left_contact = False
        right_contact = False
        try:
            for box_id in self.scene_manager.box_ids:
                contacts = p.getContactPoints(self.sys.robot_id, box_id)
                for contact in contacts:
                    link_index = contact[3]   # linkIndexA — vì bodyA = robot_id
                    if contact[8] <= 0.002:
                        if link_index in self.left_finger_links:
                            left_contact = True
                        elif link_index in self.right_finger_links:
                            right_contact = True
        except (p.error, AttributeError):
            return False, False
        return left_contact, right_contact

    def run(self):
        print("[Communication] Initializing...")
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.scene_manager.setup_pick_and_place_scene()
        # box_poses = []
        # for i, box_id in enumerate(self.scene_manager.box_ids):
        #     pos, orn_xyzw = p.getBasePositionAndOrientation(box_id)
        #     box_poses.append({
        #         "name": f"box_{i}",
        #         "position": list(pos),
        #         "quaternion_xyzw": list(orn_xyzw),
        #     })
        # self.sys.shared_box_poses.write(box_poses)
        self.sys.scene_ready.set()
        # print(f"[Communication] Synced {len(box_poses)} box poses to shared state")

        WORLD_CAM_EYE = [0.0, 1, 0.8]
        WORLD_CAM_TARGET = [0.0, 0.0, 0.4]
        yaw, pitch, distance = _eye_target_to_yaw_pitch_dist(WORLD_CAM_EYE, WORLD_CAM_TARGET)
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=WORLD_CAM_TARGET,
        )

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

        left_finger_names = ["left_inner_finger_joint", "left_inner_knuckle_joint"]
        right_finger_names = ["right_inner_finger_joint", "right_inner_knuckle_joint", "right_outer_knuckle_joint"]

        self.left_finger_links = {name_to_index[n] for n in left_finger_names if n in name_to_index}
        self.right_finger_links = {name_to_index[n] for n in right_finger_names if n in name_to_index}

        self.gripper_constraints = []
        for slave_idx, multiplier in zip(slave_indices, mimic_multipliers):
            c = p.createConstraint(
                self.sys.robot_id, master_idx,
                self.sys.robot_id, slave_idx,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],   # trục xoay của finger joint — kiểm tra đúng trục theo URDF của bạn
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=10000, erp=0.5)
            self.gripper_constraints.append(c)

        for slave_idx in slave_indices:
            p.setJointMotorControl2(self.sys.robot_id, slave_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        home_joints_deg = [0.0, 0.0, 0.0, 0.0, 90.0, 90.0]
        home_joints_rad = [math.radians(deg) for deg in home_joints_deg]

        for idx, target_rad in zip(self.sys.arm_indices, home_joints_rad):
            p.resetJointState(self.sys.robot_id, idx, target_rad)

        self.sys.shared_joints.write(home_joints_rad + [0.0], time.time())

        print("[Communication] PyBullet simulation started")
        log_counter = 0

        # Tune gripper finger contact/damping parameters to improve grasp stability
        try:
            for idx in slave_indices + [master_idx]:
                try:
                    # increase lateral friction for better hold; keep low spinning/rolling frictions
                    p.changeDynamics(self.sys.robot_id, idx, lateralFriction=1.6, spinningFriction=0.005,
                                        rollingFriction=0.0005, restitution=0.0, contactProcessingThreshold=0.001)
                except Exception:
                    print (f"[Communication] Failed to change dynamics for joint index {idx}")
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

                            t_c1 = time.perf_counter()
                            world_frame = self._get_world_camera_data()
                            # self.world_bgr = cv2.cvtColor(world_frame, cv2.COLOR_RGB2BGR)
                            t_world_ms = (time.perf_counter() - t_c1) * 1000.0

                            t_c0 = time.perf_counter()
                            wrist_frame = self._get_wrist_camera_data(tool0_link_state)
                            wrist_bgr = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
                            t_wrist_ms = (time.perf_counter() - t_c0) * 1000.0

                            # combined_frame = np.vstack((self.wrist_bgr, self.world_bgr))
                            # self.sys.shared_pybullet_frame.write(combined_frame)
                            self.sys.shared_pybullet_frame.write(wrist_bgr)

                            t_tick_total_ms = (time.perf_counter() - t_tick_start) * 1000.0

                            if self.sys.request_save_episode:
                                self.sys.dataset.save_episode()
                                self.sys.request_save_episode = False
                                print("[Record] Episode saved.")
                                
                            if self.sys.request_clear_episode:
                                self.sys.dataset.clear_episode_buffer()
                                self.sys.request_clear_episode = False
                                print("[Record] Episode discarded.")

                            # 2. Ghi dữ liệu khi hệ thống đang ở trạng thái active
                            if self.sys.teleop_active:
                                actual_master_pos = p.getJointState(self.sys.robot_id, master_idx)[0]
                                gripper_state_norm = actual_master_pos / close_limit
                                
                                self.sys.dataset.add_frame({
                                    "action": np.asarray(list(joints[:6]) + [joints[6]], dtype=np.float32),
                                    "observation.state": np.asarray(
                                        pb_state["q"] + [gripper_state_norm],
                                        dtype=np.float32,),
                                    "observation.images.wrist": wrist_frame,
                                    "observation.images.world": world_frame,
                                    "task": self.sys.task_str,
                                })

                            if hasattr(self.sys, "shared_log") and self.sys.shared_log is not None:
                                self.sys.shared_log.write_record({
                                    "actual_hz": actual_hz,
                                    "t_wrist_capture_ms": t_wrist_ms,
                                    "t_world_capture_ms": t_world_ms,
                                    "t_tick_total_ms": t_tick_total_ms,
                                    "action": joints,
                                    "pb_state": pb_state,
                                })

                            actual_period_needed = max(self.RECORD_PERIOD, (t_tick_total_ms / 1000.0) * 1.2)
                            self._next_record_deadline = now + actual_period_needed

                            # self._next_record_deadline += self.RECORD_PERIOD
                            # if self._next_record_deadline < now:
                            #     self._next_record_deadline = now + self.RECORD_PERIOD

                    except p.error:
                        pass

                if joints is not None and len(joints) >= 7:
                    arm_targets = joints[:6]
                    gripper_target = joints[6]

                    for idx, target, max_vel in zip(self.sys.arm_indices, arm_targets, arm_max_vel):
                        p.setJointMotorControl2(self.sys.robot_id, idx, p.POSITION_CONTROL,targetPosition=target, force=200, 
                                                maxVelocity=max_vel)
                        
                    actual_master_pos = p.getJointState(self.sys.robot_id, master_idx)[0]
                    close_limit = 0.71 
                    gripper_state_norm = actual_master_pos / close_limit

                    left_contact, right_contact = self._has_box_contact()
                    full_grasp = left_contact and right_contact   # chỉ True khi CẢ 2 bên cùng chạm

                    if gripper_target > 0.5 and full_grasp:
                        self.gripper_contact_phase = min(self.gripper_contact_phase + 1, 5)
                    else:
                        self.gripper_contact_phase = 0

                    target_master_pos, current_force, _ = compute_gripper_control(
                                                                current_pos=actual_master_pos,
                                                                gripper_target=gripper_target,
                                                                contact_detected=full_grasp,
                                                            )

                    # drive master toward the computed target
                    p.setJointMotorControl2(
                        self.sys.robot_id,
                        master_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target_master_pos,
                        force=current_force,
                        maxVelocity=4.0,
                    )

                    # slaves should follow the master's TARGET to avoid oscillation/lag
                    # target_slave_positions = (mimic_multipliers * target_master_pos).tolist()
                    # p.setJointMotorControlArray(
                    #     self.sys.robot_id,
                    #     slave_indices,
                    #     p.POSITION_CONTROL,
                    #     targetPositions=target_slave_positions,
                    #     forces=[current_force] * len(slave_indices),
                    #     positionGains=[0.8] * len(slave_indices),
                    # )

                    log_counter += 1
                    if self.sys.teleop_active and log_counter % 12 == 0:
                        current_pybullet_angles = [p.getJointState(self.sys.robot_id, idx)[0] for idx in self.sys.arm_indices]
                        # print(f"[PyBullet-EXEC] Target: {[round(t, 2) for t in arm_targets]} | Actual: {[round(a, 2) for a in current_pybullet_angles]}")

                loop_start = time.perf_counter()
                p.stepSimulation()
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, (1.0/240.0) - elapsed)
                time.sleep(sleep_time)

        finally:                
            print("[Communication] Closing PyBullet...")