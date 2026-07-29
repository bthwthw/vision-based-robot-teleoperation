import time
from pathlib import Path
import pybullet as p
import pybullet_data

class PyBulletSimulatorWorker:
    def __init__(self, system):
        self.sys = system

    def run(self):
        print("[Communication] Initializing...")
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        urdf_path = str(Path("assets/abb_irb1200_509_gripper/irb1200_full.urdf").resolve())
        self.sys.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

        name_to_index = {}
        for i in range(p.getNumJoints(self.sys.robot_id)):
            info = p.getJointInfo(self.sys.robot_id, i)
            name_to_index[info[1].decode("utf-8")] = i
        self.sys.tcp_link_idx = name_to_index.get("robotiq_tcp_joint", 8)
                
        print(f"[Communication] TCP - Link ID={self.sys.tcp_link_idx} ({[k for k,v in name_to_index.items() if v==self.sys.tcp_link_idx]})")

        for idx in range(9, 17):
            p.changeDynamics(self.sys.robot_id, idx, jointLowerLimit=-3.14, jointUpperLimit=3.14)

        arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.sys.arm_indices = [name_to_index[name] for name in arm_joint_names if name in name_to_index]
        
        master_idx = name_to_index["finger_joint"]
        gripper_mimic_relations = {
            "left_inner_knuckle_joint": 1.0,
            "left_inner_finger_joint": -1.0,
            "right_outer_knuckle_joint": -1.0, 
            "right_inner_knuckle_joint": -1.0, 
            "right_inner_finger_joint": 1.0,   
        }

        print("[Communication] PyBullet simulation started")

        try:
            while self.sys.is_running:
                joints, _ts = self.sys.shared_joints.read()
                if joints is not None and len(joints) >= 7:
                    arm_targets = joints[:6]
                    gripper_target = joints[6]

                    for idx, target in zip(self.sys.arm_indices, arm_targets):
                        p.setJointMotorControl2(self.sys.robot_id, idx, p.POSITION_CONTROL, targetPosition=target, force=500, maxVelocity=5.0)
                    current_pybullet_angles = [p.getJointState(self.sys.robot_id, idx)[0] for idx in self.sys.arm_indices]
                    print(f"[PyBullet-EXEC] Target: {[round(t, 2) for t in arm_targets]} | Thực tế mô phỏng: {[round(a, 2) for a in current_pybullet_angles]}")
                    p.setJointMotorControl2(self.sys.robot_id, master_idx, p.POSITION_CONTROL, targetPosition=gripper_target, force=200, maxVelocity=5.0)

                    for joint_name, multiplier in gripper_mimic_relations.items():
                        if joint_name in name_to_index:
                            slave_idx = name_to_index[joint_name]
                            p.setJointMotorControl2(self.sys.robot_id, slave_idx, p.POSITION_CONTROL, targetPosition=multiplier * gripper_target, force=200)

                p.stepSimulation()
                time.sleep(1.0 / 240.0)
        finally:
            print("[Communication] Closing ...")