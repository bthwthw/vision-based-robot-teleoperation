import time
from pathlib import Path
import numpy as np
import pybullet as p
import torch
from scipy.spatial.transform import Rotation as R

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

class cuRoboControllerWorker:
    def __init__(self, system):
        self.sys = system

    def run(self):
        print("[Controller] Initializing...")
        tensor_args = TensorDeviceType()
        robot_file = "assets/abb_irb1200_509_gripper/abb_robotiq.yml"
        
        yml_data = load_yaml(robot_file)
        kinematics_cfg = yml_data["robot_cfg"]["kinematics"]
        kinematics_cfg["urdf_path"] = str(Path(kinematics_cfg["urdf_path"]).resolve())
        kinematics_cfg["asset_root_path"] = str(Path(kinematics_cfg["asset_root_path"]).resolve())
        
        robot_cfg = RobotConfig.from_dict(yml_data["robot_cfg"], tensor_args)
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg, None, rotation_threshold=0.05, position_threshold=0.005,
            num_seeds=20, self_collision_check=False, self_collision_opt=False,
            tensor_args=tensor_args, use_cuda_graph=True,
        )
        ik_solver = IKSolver(ik_config)
        print("[Controller] cuRobo IK Solver ready")
        
        retract_q = ik_solver.robot_config.kinematics.cspace.retract_config
        retract_q = retract_q.detach().cpu().numpy()
        retract_list = np.atleast_1d(retract_q).flatten().tolist()
        
        retract_tensor = torch.tensor([retract_list], dtype=torch.float32, device=tensor_args.device)
        retract_fk = ik_solver.fk(retract_tensor)
        
        robot_base_pos = retract_fk.ee_position[0].cpu().numpy().tolist()
        robot_base_quat_wxyz = retract_fk.ee_quaternion[0].cpu().numpy().tolist()
        
        print(f"[Controller] TCP init: Pos={[round(float(p), 3) for p in robot_base_pos]}, Quat={[round(float(q), 3) for q in robot_base_quat_wxyz]}")

        print("[Controller] cuRobo Warm-up...")
        dummy_pos = torch.tensor([robot_base_pos], dtype=torch.float32, device=tensor_args.device)
        dummy_quat = torch.tensor([robot_base_quat_wxyz], dtype=torch.float32, device=tensor_args.device)
        dummy_goal = Pose(dummy_pos, dummy_quat)
        
        for _ in range(10): 
            ik_solver.solve_batch(dummy_goal)
        
        print("[Controller] cuRobo Warm-up done")
        self.sys.system_ready = True

        hand_start_pos = None                
        hand_start_rot = None  
        last_q_solution = None

        robot_start_rot = R.from_quat([robot_base_quat_wxyz[1], robot_base_quat_wxyz[2], robot_base_quat_wxyz[3], robot_base_quat_wxyz[0]])
        last_robot_target_pos = robot_base_pos
        last_robot_target_rot = robot_start_rot

        core_mapping = [
            [ 0, -1,  0],
            [ -1,  0,  0],
            [ 0,  0,  -1]
        ]
        gain = 1.5
        R_map = R.from_matrix(core_mapping)
        P_map = np.array(core_mapping) * gain
        
        try:
            while self.sys.is_running:
                pose_dict, ts = self.sys.shared_pose.read()
                if pose_dict is None or pose_dict["position"] is None:
                    time.sleep(0.005)
                    continue 

                if self.sys.request_toggle_teleop:
                    self.sys.teleop_active = not self.sys.teleop_active
                    self.sys.request_toggle_teleop = False

                    if self.sys.teleop_active:
                        raw_hand_pos = pose_dict["position"]
                        hand_start_pos = (raw_hand_pos.tolist() if isinstance(raw_hand_pos, np.ndarray) else list(raw_hand_pos))[:3]
                        
                        quat_wxyz = pose_dict["quaternion"]
                        hand_start_rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                        
                        robot_base_pos = last_robot_target_pos
                        robot_start_rot = last_robot_target_rot
                        print(f"[Controller] -> START CONTROL")
                    else:
                        print("[Controller] -> STOP CONTROL")

                if not self.sys.teleop_active:
                    if last_q_solution is not None:
                        gripper_opening = 0.725 if pose_dict["gripper"] == "Close" else 0.0
                        q_log = last_q_solution.cpu().numpy().flatten().tolist()[:6]
                        self.sys.shared_joints.write(q_log + [gripper_opening], time.time())
                    time.sleep(0.01)
                    continue
                
                raw_hand_pos = pose_dict["position"]
                raw_hand_pos = raw_hand_pos.tolist() if isinstance(raw_hand_pos, np.ndarray) else list(raw_hand_pos)
                
                quat_wxyz = pose_dict["quaternion"]
                quat_wxyz = quat_wxyz.tolist() if isinstance(quat_wxyz, np.ndarray) else list(quat_wxyz)
                current_hand_rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                
                delta_hand_cam = np.array([
                    raw_hand_pos[0] - hand_start_pos[0],
                    raw_hand_pos[1] - hand_start_pos[1],
                    raw_hand_pos[2] - hand_start_pos[2]
                ])
                
                delta_robot = P_map @ delta_hand_cam
                robot_target_pos = [robot_base_pos[0] + delta_robot[0], robot_base_pos[1] + delta_robot[1], robot_base_pos[2] + delta_robot[2]]
                
                delta_rot_cam = current_hand_rot * hand_start_rot.inv()
                delta_rot_rob = R_map * delta_rot_cam * R_map.inv()
                
                robot_start_rot = R.from_quat([robot_base_quat_wxyz[1], robot_base_quat_wxyz[2], robot_base_quat_wxyz[3], robot_base_quat_wxyz[0]])
                robot_target_rot = delta_rot_rob * robot_start_rot
                
                target_quat_xyzw = robot_target_rot.as_quat()
                robot_target_quat = [target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]]

                last_robot_target_pos = robot_target_pos
                last_robot_target_rot = robot_target_rot
                
                goal_pos_tensor = torch.tensor([robot_target_pos], dtype=torch.float32, device=tensor_args.device)
                goal_quat_tensor = torch.tensor([robot_target_quat], dtype=torch.float32, device=tensor_args.device)
                goal = Pose(goal_pos_tensor, goal_quat_tensor)
                
                seed_config = None
                if last_q_solution is not None:
                    seed_config = last_q_solution.view(1, 1, -1).repeat(1, ik_config.num_seeds, 1)

                t_start = time.time()
                result = ik_solver.solve_batch(goal, seed_config=seed_config)
                curobo_latency_ms = (time.time() - t_start) * 1000.0
                
                gripper_opening = 0.725 if pose_dict["gripper"] == "Close" else 0.0
                is_success = bool(result.success[0].item()) if hasattr(result.success[0], "item") else bool(result.success[0])
                
                q_logged = [0.0] * 6
                if is_success:
                    q_solution = result.solution[0]
                    if hasattr(q_solution, "cpu"): q_solution = q_solution.cpu().numpy()
                    q_solution = np.atleast_1d(q_solution).flatten().tolist()
                    print(f"[Controller-IK-OK] Joints (cuRobo): {[round(float(j), 3) for j in q_solution[:6]]}")
                    q_logged = q_solution[:6]
                    
                    self.sys.shared_joints.write(q_solution[:6] + [gripper_opening], time.time())
                    last_q_solution = torch.tensor(q_solution[:6], dtype=torch.float32, device=tensor_args.device)
                else:
                    print(f"[Controller-IK-FAIL] Cannot find valid joint configuration for target position: {[round(float(p), 3) for p in robot_target_pos]}")
                    last_joints, _ = self.sys.shared_joints.read()
                    if last_joints is not None and len(last_joints) >= 7:
                        q_logged = last_joints[:6]
                        self.sys.shared_joints.write(last_joints[:6] + [gripper_opening], time.time())
                
                pb_pos_logged, pb_quat_logged = [0.0]*3, [1.0, 0.0, 0.0, 0.0]
                pb_q_logged = [0.0]*6
                
                if self.sys.robot_id is not None and p.isConnected():
                    try:
                        link_state = p.getLinkState(self.sys.robot_id, self.sys.tcp_link_idx, computeForwardKinematics=True)
                        pb_pos_logged = list(link_state[4])
                        xyzw = link_state[5]
                        pb_quat_logged = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
                        pb_q_logged = [p.getJointState(self.sys.robot_id, idx)[0] for idx in self.sys.arm_indices]
                    except p.error:
                        pass
                
                ctrl_log = {
                    "frame_timestamp_s": ts, 
                    "curobo_time_ms": curobo_latency_ms,
                    "ik_success": 1 if is_success else 0,
                    "tgt_tcp_pos": robot_target_pos,
                    "tgt_tcp_quat": robot_target_quat,
                    "pb_tcp_pos": pb_pos_logged,
                    "pb_tcp_quat": pb_quat_logged,
                    "q_tgt": q_logged,
                    "pb_q": pb_q_logged
                }
                self.sys.shared_log.write_control(ctrl_log)
                time.sleep(0.01)
                
        finally:
            print("[Controller] Closing ...")