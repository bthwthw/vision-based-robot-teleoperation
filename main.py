import cv2
import math
import numpy as np
from src.module_camera import RealSenseNode
from src.module_tracker import HandTrackerNode
from src.module_hand import HandKinematics

def main():
    IS_PLAYBACK = False
    
    TCP_INDEX = None
    GRIPPER_INDEXES = [4, 8]     
    BASE_INDEXES = [0, 5, 9, 17] 
    thres = 20 # OPEN-CLOSE threshold in mm
    
    if IS_PLAYBACK:
        playback_file = r"data/20260706_151013.db3" 
    else:
        playback_file = None
        
    camera = RealSenseNode(playback_file=playback_file)
    tracker = HandTrackerNode(model_path='model/hand_landmarker.task')

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
                    cv2.putText(color_img, f"Gripper: {status}", (20, 130), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                BASE1_3D = camera.extract_3d_coordinates(landmarks[0][0], landmarks[0][1], depth_frame, depth_arr)
                BASE2_3D = camera.extract_3d_coordinates(landmarks[5][0], landmarks[5][1], depth_frame, depth_arr)
                BASE3_3D = camera.extract_3d_coordinates(landmarks[9][0], landmarks[9][1], depth_frame, depth_arr)
                BASE4_3D = camera.extract_3d_coordinates(landmarks[17][0], landmarks[17][1], depth_frame, depth_arr)

                if all(pt is not None for pt in [BASE1_3D, BASE2_3D, BASE3_3D, BASE4_3D]):

                    if TCP_INDEX is None:
                        uTCP, vTCP = int((u_gr1 + u_gr2) / 2), int((v_gr1 + v_gr2) / 2)
                        P_TCP_3D = (
                            (GR1_3D[0] + GR2_3D[0]) / 2.0,
                            (GR1_3D[1] + GR2_3D[1]) / 2.0,
                            (GR1_3D[2] + GR2_3D[2]) / 2.0
                        ) if (GR1_3D and GR2_3D) else None
                        tcp_color = (0, 165, 255)
                    else:
                        uTCP, vTCP = landmarks[TCP_INDEX]
                        P_TCP_3D = camera.extract_3d_coordinates(uTCP, vTCP, depth_frame, depth_arr)
                        tcp_color = (0, 255, 255)

                    if P_TCP_3D:
                        cv2.circle(color_img, (uTCP, vTCP), 8, tcp_color, cv2.FILLED)
                        
                        tcp_text = f"TCP Pos: X:{P_TCP_3D[0]:.3f} Y:{P_TCP_3D[1]:.3f} Z:{P_TCP_3D[2]:.3f} m"
                        cv2.putText(color_img, tcp_text, (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, tcp_color, 2, cv2.LINE_AA)

                    
                    orientation_data = HandKinematics.compute_orientation(BASE1_3D, BASE2_3D, BASE3_3D, BASE4_3D)
                    
                    if orientation_data:
                        rpy = orientation_data['rpy']
                        quat = orientation_data['quaternion']
                        
                        rpy_text = f"RPY: R:{rpy[0]:.1f} P:{rpy[1]:.1f} Y:{rpy[2]:.1f} deg"
                        cv2.putText(color_img, rpy_text, (20, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2, cv2.LINE_AA)
                        
                else:
                    cv2.putText(color_img, "[MAIN WARNING] Kinematics: Missing 3D Base Points", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            else:
                cv2.putText(color_img, "[MAIN WARNING] Cannot find hand", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

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
        cv2.destroyAllWindows()
        print("[MAIN INFO] Resources released successfully.")

if __name__ == "__main__":
    main()