import math
import time
import cv2
import numpy as np
from pathlib import Path
import pybullet as p
import pybullet_data

from src.module_scene import SceneManager


def compute_gripper_control(current_pos, gripper_target, contact_detected=False,
                           open_limit=0.0, close_limit=0.71,
                           step=0.02, close_force=28.0, open_force=30.0, hold_force=8.0,
                           contact_step=0.001):
    if gripper_target <= 0.5:
        target = max(current_pos - step, open_limit)
        force = open_force
        holding = False
    else:
        if contact_detected:
            target = min(current_pos + contact_step, close_limit)
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

    def _get_wrist_camera_frame(self, link_state, width=320, height=240):
        tcp_pos = np.array(link_state[4])
        tcp_orn_xyzw = link_state[5]
        rot_matrix = np.reshape(p.getMatrixFromQuaternion(tcp_orn_xyzw), (3, 3))

        approach_axis_local = rot_matrix[:, 2]   
        up_axis_local = rot_matrix[:, 1]         
        CAM_BACK_OFFSET = 0.10   
        CAM_FORWARD_LOOK = 0.15  

        cam_eye = tcp_pos - approach_axis_local * CAM_BACK_OFFSET + up_axis_local * 0.05
        cam_target = tcp_pos + approach_axis_local * CAM_FORWARD_LOOK

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=up_axis_local.tolist(),
        )
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.01, farVal=1.0)

        _, _, rgb_img, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3].astype(np.uint8)
        return rgb_array

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
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.scene_manager.setup_pick_and_place_scene()

        urdf_path = str(Path("assets/abb_irb1200_509_gripper/irb1200_full.urdf").resolve())
        self.sys.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

        name_to_index = {}
        for i in range(p.getNumJoints(self.sys.robot_id)):
            info = p.getJointInfo(self.sys.robot_id, i)
            name_to_index[info[1].decode("utf-8")] = i
        self.sys.tcp_link_idx = name_to_index.get("robotiq_tcp_joint", 8)
                
        print(f"[Communication] TCP - Link ID={self.sys.tcp_link_idx} ({[k for k,v in name_to_index.items() if v==self.sys.tcp_link_idx]})")

        gripper_joint_names = ["left_outer_finger_joint", "left_inner_knuckle_joint",
                               "left_inner_finger_joint", "right_outer_knuckle_joint",
                               "right_inner_knuckle_joint", "right_inner_finger_joint", "right_outer_finger_joint"] 
        for name in gripper_joint_names:
            if name in name_to_index:
                p.changeDynamics(self.sys.robot_id, name_to_index[name], jointLowerLimit=-3.14, jointUpperLimit=3.14)

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
            while self.sys.is_running:
                joints, _ts = self.sys.shared_joints.read()

                if self.sys.robot_id is not None:
                    try:
                        link_state = p.getLinkState(self.sys.robot_id, self.sys.tcp_link_idx, computeForwardKinematics=True)
                        self._draw_tcp_axes(link_state, axis_length=0.15)

                        if log_counter % 6 == 0:  # ~40Hz
                            wrist_frame = self._get_wrist_camera_frame(link_state)
                    except p.error:
                        pass

                if joints is not None and len(joints) >= 7:
                    arm_targets = joints[:6]
                    gripper_target = joints[6]

                    for idx, target, max_vel in zip(self.sys.arm_indices, arm_targets, arm_max_vel):
                        p.setJointMotorControl2(self.sys.robot_id, idx, p.POSITION_CONTROL,targetPosition=target, force=200, 
                                                maxVelocity=max_vel)
                    current_pybullet_angles = [p.getJointState(self.sys.robot_id, idx)[0] for idx in self.sys.arm_indices]
                    
                    # p.setJointMotorControl2(self.sys.robot_id, master_idx, p.POSITION_CONTROL, targetPosition=gripper_target, 
                    #                         force=50, maxVelocity=5.0)
                    actual_master_pos = p.getJointState(self.sys.robot_id, master_idx)[0]

                    # target_slave_positions = mimic_multipliers * actual_master_pos
                    # p.setJointMotorControlArray(
                    #     self.sys.robot_id, 
                    #     slave_indices, 
                    #     p.POSITION_CONTROL, 
                    #     targetPositions=target_slave_positions.tolist(),
                    #     forces=[50] * len(slave_indices), 
                    #     positionGains=np.ones(len(slave_indices)) 
                    # )

                    contact_detected = self._has_box_contact()
                    target_master_pos, current_force, _ = compute_gripper_control(
                        current_pos=actual_master_pos,
                        gripper_target=gripper_target,
                        contact_detected=contact_detected,
                    )

                    p.setJointMotorControl2(
                        self.sys.robot_id,
                        master_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target_master_pos,
                        force=current_force,
                        maxVelocity=4.0,
                    )

                    target_slave_positions = mimic_multipliers * actual_master_pos
                    p.setJointMotorControlArray(
                        self.sys.robot_id,
                        slave_indices,
                        p.POSITION_CONTROL,
                        targetPositions=target_slave_positions.tolist(),
                        forces=[current_force] * len(slave_indices),
                        positionGains=np.ones(len(slave_indices)),
                    )

                    log_counter += 1
                    if self.sys.teleop_active and log_counter % 12 == 0:
                        current_pybullet_angles = [p.getJointState(self.sys.robot_id, idx)[0] for idx in self.sys.arm_indices]
                        print(f"[PyBullet-EXEC] Target: {[round(t, 2) for t in arm_targets]} | Actual: {[round(a, 2) for a in current_pybullet_angles]}")

                p.stepSimulation()
                time.sleep(1.0 / 240.0)
        finally:
            print("[Communication] Closing ...")