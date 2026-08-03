import math
import time
import numpy as np
from pathlib import Path
import pybullet as p
import pybullet_data

from src.module_scene import SceneManager

class PyBulletSimulatorWorker:
    def __init__(self, system):
        self.sys = system
        self.axis_ids = [-1, -1, -1]
        self.scene_manager = SceneManager(table_top_z=0.4)

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

    def run(self):
        print("[Communication] Initializing...")
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.scene_manager.setup_pick_and_place_scene()

        urdf_path = str(Path("assets/abb_irb1200_509_gripper/irb1200_full.urdf").resolve())
        self.sys.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True,
                                       flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

        name_to_index = {}
        for i in range(p.getNumJoints(self.sys.robot_id)):
            info = p.getJointInfo(self.sys.robot_id, i)
            name_to_index[info[1].decode("utf-8")] = i
        self.sys.tcp_link_idx = name_to_index.get("robotiq_tcp_joint", 8)
                
        print(f"[Communication] TCP - Link ID={self.sys.tcp_link_idx} ({[k for k,v in name_to_index.items() if v==self.sys.tcp_link_idx]})")

        gripper_joint_names = ["left_outer_finger_joint", "left_inner_knuckle_joint",
                        "left_inner_finger_joint", "right_outer_knuckle_joint",
                        "right_inner_knuckle_joint", "right_inner_finger_joint", "right_outer_finger_joint"] #"finger_joint", 
        for name in gripper_joint_names:
            if name in name_to_index:
                p.changeDynamics(self.sys.robot_id, name_to_index[name], jointLowerLimit=-3.14, jointUpperLimit=3.14)

        for i in range(p.getNumJoints(self.sys.robot_id)):
            p.changeDynamics(self.sys.robot_id, i, 
                lateralFriction=2.5,     # ma sát trượt
                spinningFriction=0.05,   # ma sát xoay chống xoay hộp
                rollingFriction=0.05,    # ma sát lăn
                contactStiffness=1e4,    # độ cứng bề mặt tiếp xúc
                contactDamping=10.0      # độ giảm chấn
            )

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
        gripper_mimic_relations = {
            "left_inner_knuckle_joint": 1.0,
            "left_inner_finger_joint": -1.0,
            "right_outer_knuckle_joint": -1.0, 
            "right_inner_knuckle_joint": -1.0, 
            "right_inner_finger_joint": 1.0,   
        }

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
                    except p.error:
                        pass

                if joints is not None and len(joints) >= 7:
                    arm_targets = joints[:6]
                    gripper_target = joints[6]

                    for idx, target, max_vel in zip(self.sys.arm_indices, arm_targets, arm_max_vel):
                        p.setJointMotorControl2(self.sys.robot_id, idx, p.POSITION_CONTROL,targetPosition=target, force=200, 
                                                maxVelocity=max_vel)
                    current_pybullet_angles = [p.getJointState(self.sys.robot_id, idx)[0] for idx in self.sys.arm_indices]
                    # print(f"[PyBullet-EXEC] Target: {[round(t, 2) for t in arm_targets]} | Actual: {[round(a, 2) for a in current_pybullet_angles]}")
                    p.setJointMotorControl2(self.sys.robot_id, master_idx, p.POSITION_CONTROL, targetPosition=gripper_target, 
                                            force=20, maxVelocity=5.0)

                    # if gripper_target > 0.01:
                    #     p.setJointMotorControl2(self.sys.robot_id, master_idx, p.VELOCITY_CONTROL, targetVelocity=2.0, force=15)
                    # else:
                    #     p.setJointMotorControl2(self.sys.robot_id, master_idx, p.VELOCITY_CONTROL, targetVelocity=-2.0, force=15)

                    actual_master_pos = p.getJointState(self.sys.robot_id, master_idx)[0]

                    for joint_name, multiplier in gripper_mimic_relations.items():
                        if joint_name in name_to_index:
                            slave_idx = name_to_index[joint_name]
                            p.setJointMotorControl2(self.sys.robot_id, slave_idx, p.POSITION_CONTROL, targetPosition=multiplier * actual_master_pos, force=20, maxVelocity=5.0)

                    log_counter += 1
                    if self.sys.teleop_active and log_counter % 12 == 0:
                        current_pybullet_angles = [p.getJointState(self.sys.robot_id, idx)[0] for idx in self.sys.arm_indices]
                        print(f"[PyBullet-EXEC] Target: {[round(t, 2) for t in arm_targets]} | Actual: {[round(a, 2) for a in current_pybullet_angles]}")

                p.stepSimulation()
                time.sleep(1.0 / 240.0)
        finally:
            print("[Communication] Closing ...")