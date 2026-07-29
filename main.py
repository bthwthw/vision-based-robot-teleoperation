import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import pyrealsense2 as rs
import torch
from curobo.geom.types import WorldConfig

# cuRobo (v0.7.6)
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from scipy.spatial.transform import Rotation as R

from src.module_camera import RealSenseNode
from src.module_filter import Position3DFilter, QuaternionFilter, Scalar1DFilter
from src.module_hand import HandKinematics
from src.module_logger import DataLogger
from src.module_tracker import HandTrackerNode
from tools.analyze_filter import analyze
from tools.plot_vision import generate_report_figures


def draw_3d_axes(image, intrinsics, origin_3d, rot_matrix, axis_length=0.015):
    try:
        u0, v0 = rs.rs2_project_point_to_pixel(intrinsics, origin_3d)
        p0 = (int(u0), int(v0))
        
        x_3d = np.array(origin_3d) + rot_matrix[:, 0] * axis_length
        y_3d = np.array(origin_3d) + rot_matrix[:, 1] * axis_length
        z_3d = np.array(origin_3d) + rot_matrix[:, 2] * axis_length
        
        ux, vx = rs.rs2_project_point_to_pixel(intrinsics, x_3d.tolist())
        uy, vy = rs.rs2_project_point_to_pixel(intrinsics, y_3d.tolist())
        uz, vz = rs.rs2_project_point_to_pixel(intrinsics, z_3d.tolist())
        
        cv2.line(image, p0, (int(uy), int(vy)), (0, 255, 0), 3) # Y - Lá
        cv2.line(image, p0, (int(uz), int(vz)), (255, 0, 0), 3) # Z - Lam
        cv2.line(image, p0, (int(ux), int(vx)), (0, 0, 255), 3) # X - Đỏ
    except (RuntimeError, ValueError):
        pass
    return image

def draw_axes_legend(image):
    h, _w, _ = image.shape
    start_x = 20
    start_y = h - 120 
    
    cv2.putText(image, "Coordinate System:", (start_x, start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    
    cv2.line(image, (start_x, start_y + 55), (start_x + 30, start_y + 55), (0, 255, 0), 3)
    cv2.putText(image, "Y", (start_x + 40, start_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.line(image, (start_x, start_y + 25), (start_x + 30, start_y + 25), (0, 0, 255), 3)
    cv2.putText(image, "X", (start_x + 40, start_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.line(image, (start_x, start_y + 85), (start_x + 30, start_y + 85), (255, 0, 0), 3)
    cv2.putText(image, "Z", (start_x + 40, start_y + 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
    return image

class SharedPoseState:
    def __init__(self):
        self._lock = threading.Lock()
        self._pose = None
        self._timestamp = 0.0

    def write(self, pose, timestamp):
        with self._lock:
            self._pose = pose
            self._timestamp = timestamp

    def read(self):
        with self._lock:
            return self._pose, self._timestamp

class SharedJointState:
    def __init__(self):
        self._lock = threading.Lock()
        self._joints = None
        self._timestamp = 0.0

    def write(self, joints, timestamp):
        with self._lock:
            self._joints = joints
            self._timestamp = timestamp

    def read(self):
        with self._lock:
            return self._joints, self._timestamp

class SharedFrameState:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def write(self, frame):
        with self._lock:
            if frame is not None:
                self._frame = frame.copy()

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

class TeleopSystem:
    def __init__(self, playback_file=None):
        self.shared_pose = SharedPoseState()
        self.shared_joints = SharedJointState()
        self.shared_frame = SharedFrameState()
        
        self.log_queue = queue.Queue(maxsize=10000)
        self.is_running = False
        self.threads = []

        self.playback_file = playback_file
        self.logger_filepath = None

    def _perception_thread(self):
        print("[Perception] Initializing...")
        
        TCP_INDEX = 0
        GRIPPER_INDEXES = [3, 5]     
        BASE_INDEXES = [5,0,9,13] 
        thres_open = 45.0   # (mm) open threshold
        thres_close = 35.0  # (mm) close threshold
        gripper_state = "Open"
        
        camera = RealSenseNode(playback_file=self.playback_file)
        tracker = HandTrackerNode(model_path='model/hand_landmarker.task')
        
        tcp_filter = Position3DFilter(min_cutoff=0.1, beta=20.0, cutoff_max=15.0, reject_max_jump_mps=2.18, slew_limit_mps=1.0)
        quat_filter = QuaternionFilter(min_cutoff=0.1, beta=1.0, cutoff_max=20.0, reject_max_omega=5.08, max_rejects=10)
        gripper_filter = Scalar1DFilter(min_cutoff=1.0, beta=0.002, cutoff_max=10.0, reject_max_jump=3500.0, slew_limit_mps=4000)
        
        try:
            while self.is_running:
                color_img, depth_frame, depth_arr, timestamp = camera.get_frames()
                if color_img is None:
                    continue

                tracker.detect_async(color_img, timestamp)
                color_img = tracker.draw_skeleton(color_img)
                landmarks = tracker.get_all_landmarks_pixel(color_img)
                handedness = tracker.get_handedness(color_img)

                log_data_dict = {
                    "frame_timestamp_s": timestamp / 1000.0,
                    "raw_pos": None, "filt_pos": None,
                    "raw_quat": None, "filt_quat": None,
                    "raw_gripper_dist": None, "filt_gripper_dist": None
                }

                if landmarks:
                    u_gr1, v_gr1 = landmarks[GRIPPER_INDEXES[0]]
                    u_gr2, v_gr2 = landmarks[GRIPPER_INDEXES[1]]
                    GR1_3D = camera.extract_3d_coordinates(u_gr1, v_gr1, depth_frame, depth_arr)
                    GR2_3D = camera.extract_3d_coordinates(u_gr2, v_gr2, depth_frame, depth_arr)
                    
                    if GR1_3D and GR2_3D:
                        cv2.circle(color_img, (u_gr1, v_gr1), 8, (255, 255, 0), cv2.FILLED)                    
                        cv2.circle(color_img, (u_gr2, v_gr2), 8, (255, 255, 0), cv2.FILLED)     
                        
                        raw_dist_mm = math.sqrt((GR2_3D[0] - GR1_3D[0])**2 + (GR2_3D[1] - GR1_3D[1])**2 + (GR2_3D[2] - GR1_3D[2])**2) * 1000
                        smoothed_dist_mm = gripper_filter.filter(raw_dist_mm, timestamp / 1000.0)
                        
                        if smoothed_dist_mm > thres_open: gripper_state = "Open"
                        elif smoothed_dist_mm < thres_close: gripper_state = "Close"
                        
                        color = (0, 0, 255) if gripper_state == "Close" else (0, 255, 0)
                        cv2.putText(color_img, f"Tay kep: {gripper_state} - KC: {smoothed_dist_mm:.2f} mm", (20, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                        
                        log_data_dict["raw_gripper_dist"] = raw_dist_mm
                        log_data_dict["filt_gripper_dist"] = smoothed_dist_mm

                    BASE1_3D = camera.extract_3d_coordinates(landmarks[BASE_INDEXES[0]][0], landmarks[BASE_INDEXES[0]][1], depth_frame, depth_arr)
                    BASE2_3D = camera.extract_3d_coordinates(landmarks[BASE_INDEXES[1]][0], landmarks[BASE_INDEXES[1]][1], depth_frame, depth_arr)
                    BASE3_3D = camera.extract_3d_coordinates(landmarks[BASE_INDEXES[2]][0], landmarks[BASE_INDEXES[2]][1], depth_frame, depth_arr)
                    BASE4_3D = camera.extract_3d_coordinates(landmarks[BASE_INDEXES[3]][0], landmarks[BASE_INDEXES[3]][1], depth_frame, depth_arr)

                    if all(pt is not None for pt in [BASE1_3D, BASE2_3D, BASE3_3D, BASE4_3D]):
                        if TCP_INDEX is None:
                            uTCP, vTCP = int((u_gr1 + u_gr2) / 2), int((v_gr1 + v_gr2) / 2)
                            P_TCP_3D = ((GR1_3D[0] + GR2_3D[0]) / 2.0, (GR1_3D[1] + GR2_3D[1]) / 2.0, (GR1_3D[2] + GR2_3D[2]) / 2.0) if (GR1_3D and GR2_3D) else None
                        else:
                            uTCP, vTCP = landmarks[TCP_INDEX]
                            P_TCP_3D = camera.extract_3d_coordinates(uTCP, vTCP, depth_frame, depth_arr)

                        if P_TCP_3D:
                            log_data_dict["raw_pos"] = P_TCP_3D
                            P_TCP_3D = tcp_filter.filter(P_TCP_3D, timestamp / 1000.0)
                            log_data_dict["filt_pos"] = P_TCP_3D
                            
                            u_disp, v_disp = rs.rs2_project_point_to_pixel(camera.intrinsics, P_TCP_3D)
                            cv2.circle(color_img, (uTCP, vTCP), 8, (0, 255, 255), cv2.FILLED)
                            cv2.circle(color_img, (int(u_disp), int(v_disp)), 8, (255, 0, 255), cv2.FILLED)
                            cv2.putText(color_img, f"TCP: X:{P_TCP_3D[0]:.3f} Y:{P_TCP_3D[1]:.3f} Z:{P_TCP_3D[2]:.3f} m", (20, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                        # compute_orientation(P0, Px1, Px2, P3, handedness="Left"):
                        orientation_data = HandKinematics.compute_orientation(BASE1_3D, BASE2_3D, BASE3_3D, BASE4_3D, handedness=handedness)
                        if orientation_data:
                            raw_quat = orientation_data['quaternion']
                            quat = quat_filter.filter(raw_quat, timestamp / 1000.0)
                            
                            log_data_dict["raw_quat"] = raw_quat
                            log_data_dict["filt_quat"] = quat
                            
                            rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix() 
                            rpy = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
                            
                            cv2.putText(color_img, f"RPY: R:{rpy[0]:.1f} P:{rpy[1]:.1f} Y:{rpy[2]:.1f} do", (20, 130), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(color_img, f"Quat: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]", (20, 160), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                            
                            if P_TCP_3D:
                                color_img = draw_3d_axes(color_img, camera.intrinsics, P_TCP_3D, rot_matrix)
                            
                            # Ghi TCP Pose vao SharedState
                            pose_data = {"position": P_TCP_3D, "quaternion": quat, "gripper": gripper_state}
                            self.shared_pose.write(pose_data, timestamp / 1000.0)
                            
                    else:
                        cv2.putText(color_img, "[Perception WARN]: NO BASE POINTS", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(color_img, "[Perception WARN]: NO HAND DETECTED", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    tcp_filter.reset()
                    quat_filter.reset()
                    gripper_filter.reset()

                self.log_queue.put(log_data_dict)

                color_img = draw_axes_legend(color_img)
                # depth_colormap = camera.colorize_depth(depth_frame)
                # if depth_colormap.shape[1] != color_img.shape[1]:
                #     scale = color_img.shape[1] / depth_colormap.shape[1]
                #     depth_colormap = cv2.resize(depth_colormap, (color_img.shape[1], int(depth_colormap.shape[0] * scale)))
                    
                # combined_view = np.vstack((color_img, depth_colormap))
                self.shared_frame.write(color_img)

        finally:
            print("[Perception] Closing ...")
            tracker.close()
            camera.stop()

    def _controller_thread(self):
        print("[Controller] Initializing...")
        
        tensor_args = TensorDeviceType()
        robot_file = "assets/abb_irb1200_509_gripper/abb_robotiq.yml"
        
        yml_data = load_yaml(robot_file)
        kinematics_cfg = yml_data["robot_cfg"]["kinematics"]
        
        kinematics_cfg["urdf_path"] = str(Path(kinematics_cfg["urdf_path"]).resolve())
        kinematics_cfg["asset_root_path"] = str(Path(kinematics_cfg["asset_root_path"]).resolve())
        
        robot_cfg = RobotConfig.from_dict(yml_data["robot_cfg"], tensor_args)
        
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20, 
            self_collision_check=False, 
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=True,
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
        
        print(f"[Controller] TCP Gốc: Pos={[round(float(p), 3) for p in robot_base_pos]}")
        
        hand_start_pos = None               
        hand_start_rot = None  
        last_q_solution = None

        core_mapping = [
            [ 0, -1,  0],
            [ -1,  0,  0],
            [ 0,  0,  -1]
        ]
        gain = 1.5

        # Ori 
        R_map = R.from_matrix(core_mapping)
        
        try:
            while self.is_running:
                pose_dict, ts = self.shared_pose.read()
                
                if pose_dict is None or pose_dict["position"] is None:
                    time.sleep(0.005)
                    continue 
                
                raw_hand_pos = pose_dict["position"]
                raw_hand_pos = raw_hand_pos.tolist() if isinstance(raw_hand_pos, np.ndarray) else list(raw_hand_pos)
                
                quat_wxyz = pose_dict["quaternion"]
                quat_wxyz = quat_wxyz.tolist() if isinstance(quat_wxyz, np.ndarray) else list(quat_wxyz)
                # Chuyển đổi quaternion sang thư viện Scipy (định dạng xyzw)
                current_hand_rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                
                if hand_start_pos is None:
                    hand_start_pos = [raw_hand_pos[0], raw_hand_pos[1], raw_hand_pos[2]]
                    hand_start_rot = current_hand_rot
                    print(f"[Controller] Đã khóa Pose gốc của Tay thành công tại: {hand_start_pos}")
                
                delta_hand_x = raw_hand_pos[0] - hand_start_pos[0]
                delta_hand_y = raw_hand_pos[1] - hand_start_pos[1]
                delta_hand_z = raw_hand_pos[2] - hand_start_pos[2]

                P_map = np.array(core_mapping) * gain
    
                delta_hand = np.array([delta_hand_x, delta_hand_y, delta_hand_z])
                robot_target_pos = (np.array(robot_base_pos) + P_map @ delta_hand).tolist()
                
                delta_rot_cam = current_hand_rot * hand_start_rot.inv()
                delta_rot_rob = R_map * delta_rot_cam * R_map.inv()
                
                robot_start_rot = R.from_quat([
                    robot_base_quat_wxyz[1], 
                    robot_base_quat_wxyz[2], 
                    robot_base_quat_wxyz[3], 
                    robot_base_quat_wxyz[0]
                ])
                robot_target_rot = delta_rot_rob * robot_start_rot
                
                target_quat_xyzw = robot_target_rot.as_quat()
                robot_target_quat = [target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]]
                
                goal_pos_tensor = torch.tensor([robot_target_pos], dtype=torch.float32, device=tensor_args.device)
                goal_quat_tensor = torch.tensor([robot_target_quat], dtype=torch.float32, device=tensor_args.device)
                
                goal = Pose(goal_pos_tensor, goal_quat_tensor)
                
                seed_config = None
                if last_q_solution is not None:
                    seed_config = last_q_solution.view(1, 1, -1).repeat(1, ik_config.num_seeds, 1)

                result = ik_solver.solve_batch(goal, seed_config=seed_config)
                
                gripper_opening = 0.725 if pose_dict["gripper"] == "Close" else 0.0
                is_success = bool(result.success[0].item()) if hasattr(result.success[0], "item") else bool(result.success[0])
                
                if is_success:
                    q_solution = result.solution[0]
                    if hasattr(q_solution, "cpu"):
                        q_solution = q_solution.cpu().numpy()
                    q_solution = np.atleast_1d(q_solution).flatten().tolist()
                    
                    print(f"[Controller-IK-OK] Joints (cuRobo): {[round(float(j), 3) for j in q_solution[:6]]}")
                    
                    final_joints = q_solution[:6] + [gripper_opening]
                    self.shared_joints.write(final_joints, time.time())

                    last_q_solution = torch.tensor(q_solution[:6], dtype=torch.float32, device=tensor_args.device)
                    
                else:
                    print(f"[Controller-IK-FAIL] Không tìm được hướng Pose phù hợp cho đích: {[round(float(p), 3) for p in robot_target_pos]}")
                    last_joints, _ = self.shared_joints.read()
                    if last_joints is not None and len(last_joints) >= 7:
                        final_joints = last_joints[:6] + [gripper_opening]
                        self.shared_joints.write(final_joints, time.time())
                
                time.sleep(0.01)
                
        finally:
            print("[Controller] Closing ...")

    def _communication_thread(self):
        print("[Communication] Initializing...")
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        urdf_path = str(Path("assets/abb_irb1200_509_gripper/irb1200_full.urdf").resolve())
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

        name_to_index = {}
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            name_to_index[info[1].decode("utf-8")] = i

        for idx in range(9, 17):
            p.changeDynamics(robot_id, idx, jointLowerLimit=-3.14, jointUpperLimit=3.14)

        arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        arm_indices = [name_to_index[name] for name in arm_joint_names if name in name_to_index]
        
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
            while self.is_running:
                joints, _ts = self.shared_joints.read()
                
                if joints is not None and len(joints) >= 7:
                    arm_targets = joints[:6]
                    gripper_target = joints[6]

                    for idx, target in zip(arm_indices, arm_targets):
                        p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, targetPosition=target, force=500, maxVelocity=5.0)
                    
                    current_pybullet_angles = [p.getJointState(robot_id, idx)[0] for idx in arm_indices]
                    print(f"[PyBullet-EXEC] Target: {[round(t, 2) for t in arm_targets]} | Thực tế mô phỏng: {[round(a, 2) for a in current_pybullet_angles]}")

                    p.setJointMotorControl2(robot_id, master_idx, p.POSITION_CONTROL, targetPosition=gripper_target, force=200, maxVelocity=5.0)

                    for joint_name, multiplier in gripper_mimic_relations.items():
                        if joint_name in name_to_index:
                            slave_idx = name_to_index[joint_name]
                            target = multiplier * gripper_target
                            p.setJointMotorControl2(robot_id, slave_idx, p.POSITION_CONTROL, targetPosition=target, force=200)

                p.stepSimulation()
                time.sleep(1.0 / 240.0)
        finally:
            print("[Communication] Closing ...")

    def _logger_thread(self):
        print("[Logger] Initializing...")
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        mode_prefix = "PB" if self.playback_file else "RT"
        log_filename = f"{mode_prefix}_{current_time_str}.csv"
        
        logger = DataLogger(log_filename)
        self.logger_filepath = logger.filepath
        
        try:
            while self.is_running or not self.log_queue.empty():
                try:
                    log_data = self.log_queue.get(timeout=0.2)
                    logger.log(**log_data)
                    self.log_queue.task_done()
                except queue.Empty:
                    continue
        finally:
            print("[Logger] Stop logging ...")
            logger.close()

    def start_all(self):
        self.is_running = True
        self.threads = [
            threading.Thread(target=self._perception_thread, name="PerceptionTh", daemon=True),
            threading.Thread(target=self._controller_thread, name="ControllerTh", daemon=True),
            threading.Thread(target=self._communication_thread, name="EGMTh", daemon=True),
            threading.Thread(target=self._logger_thread, name="LoggerTh", daemon=True)
        ]
        for t in self.threads:
            t.start()

    def stop_all(self):
        print("[System] Closing ...")
        self.is_running = False
        for t in self.threads:
            t.join() 
        print("[System] All threads stopped.")
        
        if self.logger_filepath:
            print("[System] Analyzing log data...")
            analyze(self.logger_filepath)        
            generate_report_figures(self.logger_filepath, out_dir="figs")


def main():
    system = TeleopSystem(playback_file=None )
    system.start_all()
    
    cv2.namedWindow("Teleoperation Pipeline", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            display_frame = system.shared_frame.read()
            if display_frame is not None:
                cv2.imshow("Teleoperation Pipeline", display_frame)
                
            if cv2.waitKey(33) & 0xFF == 27: 
                break
                
    except KeyboardInterrupt:
        print("[SYSTEM] KeyboardInterrupt")
    finally:
        cv2.destroyAllWindows()
        system.stop_all()

if __name__ == "__main__":
    main()