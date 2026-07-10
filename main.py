from datetime import datetime
import cv2
import math
import numpy as np
import pyrealsense2 as rs 
from scipy.spatial.transform import Rotation as R

from src.module_camera import RealSenseNode
from src.module_tracker import HandTrackerNode
from src.module_hand import HandKinematics 
from src.module_filter import Position3DFilter, QuaternionFilter
from src.module_logger import DataLogger
from tools.analyze_filter import analyze

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
        
        cv2.line(image, p0, (int(ux), int(vx)), (0, 0, 255), 3) # X - Đỏ
        cv2.line(image, p0, (int(uy), int(vy)), (0, 255, 0), 3) # Y - Lá
        cv2.line(image, p0, (int(uz), int(vz)), (255, 0, 0), 3) # Z - Lam
    except Exception as e:
        pass
    return image

def draw_axes_legend(image):
    h, w, _ = image.shape
    start_x = 20
    start_y = h - 120 
    
    cv2.putText(image, "Coordinate System:", (start_x, start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    
    cv2.line(image, (start_x, start_y + 25), (start_x + 30, start_y + 25), (0, 0, 255), 3)
    cv2.putText(image, "X", (start_x + 40, start_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.line(image, (start_x, start_y + 55), (start_x + 30, start_y + 55), (0, 255, 0), 3)
    cv2.putText(image, "Y", (start_x + 40, start_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.line(image, (start_x, start_y + 85), (start_x + 30, start_y + 85), (255, 0, 0), 3)
    cv2.putText(image, "Z", (start_x + 40, start_y + 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
    return image

def main():
    IS_PLAYBACK = False # True: playback from file, False: live stream from camera
    
    TCP_INDEX = None
    GRIPPER_INDEXES = [4, 8]     
    BASE_INDEXES = [0,1,2,5] 
    thres = 20 # open-close threshold in mm
    
    if IS_PLAYBACK:
        playback_file = r"data/20260706_151013.db3" 
    else:
        playback_file = None
        
    camera = RealSenseNode(playback_file=playback_file)
    tracker = HandTrackerNode(model_path='model/hand_landmarker.task')
    tcp_filter = Position3DFilter(min_cutoff=0.5, beta=0)
    quat_filter = QuaternionFilter(min_cutoff=1.5, beta=5)
    
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_prefix = "PB" if IS_PLAYBACK else "RT"
    log_filename = f"{mode_prefix}_{current_time_str}.csv"
    logger = DataLogger(log_filename)

    cv2.namedWindow("Teleoperation Pipeline", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Teleoperation Pipeline", 1000, 900)

    print("[MAIN INFO] Entering main execution loop...")
    
    try:
        while True:
            color_img, depth_frame, depth_arr, timestamp = camera.get_frames()
            
            if color_img is None:
                continue

            tracker.detect_async(color_img, timestamp)
            color_img = tracker.draw_skeleton(color_img)
            landmarks = tracker.get_all_landmarks_pixel(color_img)

            if landmarks:
                u_gr1, v_gr1 = landmarks[GRIPPER_INDEXES[0]]
                u_gr2, v_gr2 = landmarks[GRIPPER_INDEXES[1]]

                GR1_3D = camera.extract_3d_coordinates(u_gr1, v_gr1, depth_frame, depth_arr)
                GR2_3D = camera.extract_3d_coordinates(u_gr2, v_gr2, depth_frame, depth_arr)
                
                if GR1_3D and GR2_3D:
                    cv2.circle(color_img, (u_gr1, v_gr1), 8, (255, 255, 0), cv2.FILLED)                    
                    cv2.circle(color_img, (u_gr2, v_gr2), 8, (255, 255, 0), cv2.FILLED)     
                    
                    dist_3d_mm = math.sqrt((GR2_3D[0] - GR1_3D[0])**2 + (GR2_3D[1] - GR1_3D[1])**2 + (GR2_3D[2] - GR1_3D[2])**2) * 1000
                    status = "Close" if dist_3d_mm < thres else "Open"
                    color = (0, 0, 255) if dist_3d_mm < thres else (0, 255, 0)
                    cv2.putText(color_img, f"Gripper: {status}", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                BASE1_3D = camera.extract_3d_coordinates(landmarks[BASE_INDEXES[0]][0], landmarks[BASE_INDEXES[0]][1], depth_frame, depth_arr)
                BASE2_3D = camera.extract_3d_coordinates(landmarks[BASE_INDEXES[1]][0], landmarks[BASE_INDEXES[1]][1], depth_frame, depth_arr)
                BASE3_3D = camera.extract_3d_coordinates(landmarks[BASE_INDEXES[2]][0], landmarks[BASE_INDEXES[2]][1], depth_frame, depth_arr)
                BASE4_3D = camera.extract_3d_coordinates(landmarks[BASE_INDEXES[3]][0], landmarks[BASE_INDEXES[3]][1], depth_frame, depth_arr)

                if all(pt is not None for pt in [BASE1_3D, BASE2_3D, BASE3_3D, BASE4_3D]):

                    if TCP_INDEX is None:
                        uTCP, vTCP = int((u_gr1 + u_gr2) / 2), int((v_gr1 + v_gr2) / 2)
                        P_TCP_3D = (
                            (GR1_3D[0] + GR2_3D[0]) / 2.0,
                            (GR1_3D[1] + GR2_3D[1]) / 2.0,
                            (GR1_3D[2] + GR2_3D[2]) / 2.0
                        ) if (GR1_3D and GR2_3D) else None
                    else:
                        uTCP, vTCP = landmarks[TCP_INDEX]
                        P_TCP_3D = camera.extract_3d_coordinates(uTCP, vTCP, depth_frame, depth_arr)

                    if P_TCP_3D:
                        raw_P_TCP_3D = P_TCP_3D
                        P_TCP_3D = tcp_filter.filter(P_TCP_3D, timestamp / 1000.0)
                        u_disp, v_disp = rs.rs2_project_point_to_pixel(camera.intrinsics, P_TCP_3D)
                        cv2.circle(color_img, (uTCP, vTCP), 8, (0, 255, 255), cv2.FILLED)
                        cv2.circle(color_img, (int(u_disp), int(v_disp)), 8, (0, 0, 255), cv2.FILLED)
                        
                        tcp_text = f"TCP Pos: X:{P_TCP_3D[0]:.3f} Y:{P_TCP_3D[1]:.3f} Z:{P_TCP_3D[2]:.3f} m"
                        cv2.putText(color_img, tcp_text, (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                    orientation_data = HandKinematics.compute_orientation(BASE1_3D, BASE2_3D, BASE3_3D, BASE4_3D)
                    
                    if orientation_data:
                        # rot_matrix = orientation_data['matrix']
                        # rpy = orientation_data['rpy']
                        # quat = orientation_data['quaternion'] 
                        raw_quat = orientation_data['quaternion']
                        
                        quat = quat_filter.filter(orientation_data['quaternion'], timestamp / 1000.0)
                        rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()  # convert lại wxyz -> xyzw cho scipy
                        rpy = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
                        
                        rpy_text = f"RPY: R:{rpy[0]:.1f} P:{rpy[1]:.1f} Y:{rpy[2]:.1f} deg"
                        cv2.putText(color_img, rpy_text, (20, 130), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                                    
                        quat_text = f"Quat (w,x,y,z): [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]"
                        cv2.putText(color_img, quat_text, (20, 160), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                        
                        if P_TCP_3D:
                            color_img = draw_3d_axes(color_img, camera.intrinsics, P_TCP_3D, rot_matrix)
                        
                        logger.log(
                            frame_timestamp_s=timestamp / 1000.0,
                            raw_pos=raw_P_TCP_3D, filt_pos=P_TCP_3D,
                            raw_quat=raw_quat, filt_quat=quat,
                            gripper_dist_mm=dist_3d_mm if GR1_3D and GR2_3D else None,
                        )
                        
                else:
                    cv2.putText(color_img, "Kinematics: Missing 3D Base Points", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            else:
                cv2.putText(color_img, "[MAIN Warning] Cannot find hand", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                tcp_filter.reset()
                quat_filter.reset()

            color_img = draw_axes_legend(color_img)

            depth_colormap = camera.colorize_depth(depth_frame)
            if depth_colormap.shape[1] != color_img.shape[1]:
                scale = color_img.shape[1] / depth_colormap.shape[1]
                depth_colormap = cv2.resize(depth_colormap, (color_img.shape[1], int(depth_colormap.shape[0] * scale)))
                
            combined_view = np.vstack((color_img, depth_colormap))
            cv2.imshow("Teleoperation Pipeline", combined_view)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        print(f"[MAIN ERROR] {e}")
    finally:
        tracker.close()
        camera.stop()
        logger.close()
        cv2.destroyAllWindows()
        print("[MAIN INFO] Resources released successfully.")
        analyze(logger.filepath)

if __name__ == "__main__":
    main()