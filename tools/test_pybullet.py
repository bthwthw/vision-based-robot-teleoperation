import time
from pathlib import Path

import pybullet as p
import pybullet_data

URDF_PATH = Path("assets/abb_irb1200_509_gripper/irb1200_full.urdf")
# URDF_PATH = Path("assets/abb_irb1200_509_gripper/irb1200.urdf")


def get_joint_name_to_index(robot_id: int) -> dict:
    mapping = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        mapping[info[1].decode("utf-8")] = i
    return mapping


def print_all_joints(robot_id: int) -> None:
    print(f"\n{'Index':<6}{'Name':<28}{'Type':<12}{'Lower':<10}{'Upper':<10}")
    print("-" * 68)
    joint_type_map = {
        p.JOINT_REVOLUTE: "REVOLUTE",
        p.JOINT_PRISMATIC: "PRISMATIC",
        p.JOINT_FIXED: "FIXED",
    }
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        jtype = joint_type_map.get(info[2], str(info[2]))
        lower, upper = info[8], info[9]
        print(f"{i:<6}{name:<28}{jtype:<12}{lower:<10.3f}{upper:<10.3f}")
    print("-" * 68)


def print_link_bounding_boxes(robot_id: int) -> None:
    print(f"\n{'LinkIdx':<9}{'LinkName':<28}{'SizeX(m)':<11}{'SizeY(m)':<11}{'SizeZ(m)':<11}")
    print("-" * 75)
    num_joints = p.getNumJoints(robot_id)
    for link_idx in range(-1, num_joints):
        aabb_min, aabb_max = p.getAABB(robot_id, link_idx)
        size = [aabb_max[i] - aabb_min[i] for i in range(3)]
        if link_idx == -1:
            name = "(base_link - root)"
        else:
            info = p.getJointInfo(robot_id, link_idx)
            name = info[12].decode("utf-8")
        flag = "  <-- QUA LON, NGHI SAI SCALE!" if max(size) > 3.0 else ""
        print(f"{link_idx:<9}{name:<28}{size[0]:<11.3f}{size[1]:<11.3f}{size[2]:<11.3f}{flag}")
    print("-" * 75)

def main() -> None:
    if not URDF_PATH.exists():
        print(f"[LOI] Khong tim thay URDF: {URDF_PATH.resolve()}")
        return

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    print(f"Dang nap: {URDF_PATH.resolve()}")
    robot_id = p.loadURDF(str(URDF_PATH), basePosition=[0, 0, 0], useFixedBase=True)
    name_to_index = get_joint_name_to_index(robot_id)
    
    for idx in range(9, 17): # Từ khớp 9 đến 16
        p.changeDynamics(robot_id, idx, jointLowerLimit=-3.14, jointUpperLimit=3.14)
    print("\n[HỆ THỐNG] Đã bẻ khóa giới hạn góc âm cho Gripper!")

    # THIẾT LẬP SLIDER
    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    controllable_joints = []
    slider_ids = []
    for name in arm_joint_names:
        if name in name_to_index:
            idx = name_to_index[name]
            info = p.getJointInfo(robot_id, idx)
            slider_id = p.addUserDebugParameter(name, info[8], info[9], 0.0)
            controllable_joints.append(idx)
            slider_ids.append(slider_id)
            
    master_joint_name = "finger_joint"
    master_idx = name_to_index[master_joint_name]
    master_info = p.getJointInfo(robot_id, master_idx)
    gripper_slider_id = p.addUserDebugParameter("gripper_opening", master_info[8], master_info[9], master_info[8])
    
    gripper_mimic_relations = {
        "left_inner_knuckle_joint": 1.0,  
        "left_inner_finger_joint": -1.0,   
        
        "right_outer_knuckle_joint": -1.0, 
        "right_inner_knuckle_joint": -1.0, 
        "right_inner_finger_joint": 1.0,  
    }

    try:
        while p.isConnected():
            # Điều khiển tay máy
            for joint_idx, slider_id in zip(controllable_joints, slider_ids):
                target = p.readUserDebugParameter(slider_id)
                p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=target, force=500)

            # Điều khiển Gripper
            opening = p.readUserDebugParameter(gripper_slider_id)
            
            # Ép khớp chủ
            p.setJointMotorControl2(robot_id, master_idx, p.POSITION_CONTROL, targetPosition=opening, force=200)
            
            # Ép các khớp phụ theo hệ số
            for joint_name, multiplier in gripper_mimic_relations.items():
                if joint_name in name_to_index:
                    idx = name_to_index[joint_name]
                    target = multiplier * opening
                    p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, targetPosition=target, force=200)

            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            
    except KeyboardInterrupt:
        print("\nĐã dừng hệ thống.")
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()