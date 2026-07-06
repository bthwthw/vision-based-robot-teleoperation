import cv2
import math
import numpy as np
from src.module_camera import RealSenseNode
from src.module_tracker import HandTrackerNode

def main():
    IS_PLAYBACK = True
    if IS_PLAYBACK:
        playback_file = r"data/20260706_151013.db3" 
    else:
        playback_file = None
    camera = RealSenseNode(playback_file=playback_file)
    tracker = HandTrackerNode(model_path='model/hand_landmarker.task')

    cv2.namedWindow("Teleoperation Pipeline", cv2.WINDOW_NORMAL)

    print("[INFO] Entering main execution loop...")
    
    try:
        while True:
            color_img, depth_frame, depth_arr, timestamp = camera.get_frames()
            
            if color_img is None:
                continue

            # keypoint detection
            tracker.detect_async(color_img, timestamp)

            # draw skeleton and get keypoints
            color_img = tracker.draw_skeleton(color_img)
            keypoints = tracker.get_keypoints_pixel(color_img)

            if keypoints:
                # TCP coordinate 
                P9_3D = camera.extract_3d_coordinates(
                    keypoints['P9'][0], keypoints['P9'][1], depth_frame, depth_arr
                )
                
                # visualize tcp 
                if P9_3D:
                    u9, v9 = keypoints['P9']
                    cv2.circle(color_img, (u9, v9), 8, (0, 255, 255), cv2.FILLED)
                    cv2.putText(color_img, "TCP (P9)", (u9 + 10, v9 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    tcp_text = f"TCP (X,Y,Z): {P9_3D[0]:.3f}, {P9_3D[1]:.3f}, {P9_3D[2]:.3f} m"
                    cv2.putText(color_img, tcp_text, (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(color_img, "TCP: No depth data (Z=0)", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                # gripper distance calculation
                P4_3D = camera.extract_3d_coordinates(keypoints['P4'][0], keypoints['P4'][1], depth_frame, depth_arr)
                P8_3D = camera.extract_3d_coordinates(keypoints['P8'][0], keypoints['P8'][1], depth_frame, depth_arr)
                
                if P4_3D and P8_3D:
                    thres = 20 # mm
                    dist_3d_mm = math.sqrt((P8_3D[0] - P4_3D[0])**2 + (P8_3D[1] - P4_3D[1])**2 + (P8_3D[2] - P4_3D[2])**2) * 1000
                    status = "Close" if dist_3d_mm < thres else "Open"
                    color = (0, 0, 255) if dist_3d_mm < thres else (0, 255, 0)
                    cv2.putText(color_img, f"Status: {status}", (20, 130), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                

            else:
                cv2.putText(color_img, "[Warning] Cannot find hand", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # 5. Xử lý bản đồ nhiệt và hiển thị
            depth_colormap = camera.colorize_depth(depth_frame)
            
            # Khớp tỷ lệ hai ảnh để nối dọc
            if depth_colormap.shape[1] != color_img.shape[1]:
                scale = color_img.shape[1] / depth_colormap.shape[1]
                depth_colormap = cv2.resize(depth_colormap, (color_img.shape[1], int(depth_colormap.shape[0] * scale)))
                
            combined_view = np.vstack((color_img, depth_colormap))
            cv2.imshow("Teleoperation Pipeline", combined_view)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        tracker.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Resources released successfully.")

if __name__ == "__main__":
    main()