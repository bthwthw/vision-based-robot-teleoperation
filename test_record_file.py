import os
import cv2
import numpy as np
import math
import pyrealsense2 as rs

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ==========================================================
# Asynchronous Callback
# ==========================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),         # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),    # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ngón áp út
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) # Ngón út và lòng bàn tay
]

latest_hand_result = None

def hand_detection_callback(result, output_image, timestamp_ms):
    global latest_hand_result
    latest_hand_result = result

# ==========================================================
# MediaPipe Tasks API Initialization (LIVE_STREAM Mode)
# ==========================================
def create_mediapipe_tracker(model_path='hand_landmarker.task'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Cannot find file '{model_path}'.")

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=hand_detection_callback
    )
    return vision.HandLandmarker.create_from_options(options)

# ==========================================================
# Camera Creation & RealSense Post-Processing Filters
# ==========================================
def create_playback_pipeline(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(file_path, repeat_playback=True)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(True)
    return pipeline, profile

def create_real_camera_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    return pipeline, profile

def create_post_processing_filters():
    decimation = rs.decimation_filter()
    hdr_merge = rs.hdr_merge()
    depth_to_disparity = rs.disparity_transform(True)
    
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    disparity_to_depth = rs.disparity_transform(False)
    
    return decimation, hdr_merge, depth_to_disparity, spatial, temporal, disparity_to_depth

# ==========================================================
# Alignment & Spatial Extraction
# ==========================================
def find_nearest_valid_depth_pnt(x, y, depth_array, radius=15):
    h, w = depth_array.shape
    x_int, y_int = int(x), int(y)
    
    if not (0 <= x_int < w and 0 <= y_int < h):
        return None
        
    depth_val = depth_array[y_int, x_int]
    if depth_val > 0:
        return (x_int, y_int)
    
    y_min, y_max = max(0, y_int - radius), min(h, y_int + radius + 1)
    x_min, x_max = max(0, x_int - radius), min(w, x_int + radius + 1)
    
    depth_region = depth_array[y_min:y_max, x_min:x_max]
    
    valid_y_rel, valid_x_rel = np.where(depth_region > 0)
    
    if len(valid_y_rel) == 0:
        return None
    
    valid_x_abs = x_min + valid_x_rel
    valid_y_abs = y_min + valid_y_rel
    
    distances_sq = (valid_x_abs - x)**2 + (valid_y_abs - y)**2
    
    radius_sq = radius ** 2
    in_radius_mask = distances_sq <= radius_sq
    
    if not np.any(in_radius_mask):
        return None
        
    valid_distances = distances_sq[in_radius_mask]
    min_idx_in_mask = np.argmin(valid_distances)
    final_idx = np.where(in_radius_mask)[0][min_idx_in_mask]
    
    best_x = int(valid_x_abs[final_idx])
    best_y = int(valid_y_abs[final_idx])
    
    return (best_x, best_y)

def extract_3d_coordinates(u, v, depth_frame, depth_array, color_intrinsics):
    valid_pixel = find_nearest_valid_depth_pnt(u, v, depth_array, radius=20)
    if valid_pixel is None:
        return None
        
    best_u, best_v = valid_pixel

    depth_value = depth_frame.get_distance(best_u, best_v)
    if depth_value <= 0.0:
        return None
        
    return rs.rs2_deproject_pixel_to_point(color_intrinsics, [best_u, best_v], depth_value)

# ==========================================================
# Application Loop 
# ==========================================
def run_vision_pipeline(pipeline, profile):
    global latest_hand_result

    aligner = rs.align(rs.stream.color)
    hand_tracker = create_mediapipe_tracker()
    
    dec_filter, hdr_filter, d2d, spatial, temporal, d2d_inv = create_post_processing_filters()
    
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    colorizer = rs.colorizer()

    cv2.namedWindow("Vision Teleoperation Pipeline", cv2.WINDOW_NORMAL)

    last_hw_timestamp = 0
    timestamp_offset = 0
    last_mp_timestamp = 0

    while True:
        success, frames = pipeline.try_wait_for_frames(timeout_ms=5000)
        if not success:
            print("[INFO] No more frames or timeout.")
            break

        aligned_frames = aligner.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # post-processing depth frame
        filtered_depth = depth_frame
        filtered_depth = d2d.process(filtered_depth)
        filtered_depth = spatial.process(filtered_depth)
        filtered_depth = temporal.process(filtered_depth)
        filtered_depth = d2d_inv.process(filtered_depth).as_depth_frame()

        depth_array = np.asanyarray(filtered_depth.get_data())

        color_image_raw = np.asanyarray(color_frame.get_data())
        fmt = color_frame.profile.format()
        
        if fmt == rs.format.rgb8:
            color_image = cv2.cvtColor(color_image_raw, cv2.COLOR_RGB2BGR)
        elif fmt == rs.format.rgba8:
            color_image = cv2.cvtColor(color_image_raw, cv2.COLOR_RGBA2BGR)
        elif fmt == rs.format.bgra8:
            color_image = cv2.cvtColor(color_image_raw, cv2.COLOR_BGRA2BGR)
        else:
            color_image = np.copy(color_image_raw) 

        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image_rgb)
        
        current_hw_timestamp = int(color_frame.get_timestamp())

        if current_hw_timestamp < last_hw_timestamp:
            timestamp_offset += last_hw_timestamp
        last_hw_timestamp = current_hw_timestamp

        mp_timestamp_ms = current_hw_timestamp + timestamp_offset

        if mp_timestamp_ms <= last_mp_timestamp:
            mp_timestamp_ms = last_mp_timestamp + 1
        last_mp_timestamp = mp_timestamp_ms

        hand_tracker.detect_async(mp_image, mp_timestamp_ms)

        if latest_hand_result and latest_hand_result.hand_landmarks:
            for hand_landmarks_list in latest_hand_result.hand_landmarks:
                h, w, _ = color_image.shape
                
                u0, v0 = int(hand_landmarks_list[0].x * w), int(hand_landmarks_list[0].y * h)
                u4, v4 = int(hand_landmarks_list[4].x * w), int(hand_landmarks_list[4].y * h)
                u5, v5 = int(hand_landmarks_list[5].x * w), int(hand_landmarks_list[5].y * h)
                u8, v8 = int(hand_landmarks_list[8].x * w), int(hand_landmarks_list[8].y * h)
                u9, v9 = int(hand_landmarks_list[9].x * w), int(hand_landmarks_list[9].y * h)
                
                # landmarks drawing 
                pixel_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks_list]
                for connection in HAND_CONNECTIONS:
                    cv2.line(color_image, pixel_landmarks[connection[0]], pixel_landmarks[connection[1]], (200, 200, 200), 2)
                for pt in pixel_landmarks:
                    cv2.circle(color_image, pt, 4, (0, 215, 255), -1)

                cv2.circle(color_image, (u4, v4), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(color_image, (u8, v8), 10, (0, 0, 255), cv2.FILLED)
                cv2.line(color_image, (u4, v4), (u8, v8), (0, 255, 0), 3)

                # Truyền thêm depth_array vào hàm extract
                P0_3D = extract_3d_coordinates(u0, v0, filtered_depth, depth_array, color_intrinsics)
                P4_3D = extract_3d_coordinates(u4, v4, filtered_depth, depth_array, color_intrinsics)
                P5_3D = extract_3d_coordinates(u5, v5, filtered_depth, depth_array, color_intrinsics)
                P8_3D = extract_3d_coordinates(u8, v8, filtered_depth, depth_array, color_intrinsics)
                P9_3D = extract_3d_coordinates(u9, v9, filtered_depth, depth_array, color_intrinsics)
                
                if P9_3D:
                    cv2.circle(color_image, (u9, v9), 8, (0, 255, 255), cv2.FILLED)
                    cv2.putText(color_image, "TCP (P9)", (u9 + 10, v9 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    tcp_text = f"TCP 3D (X,Y,Z): {P9_3D[0]:.3f}, {P9_3D[1]:.3f}, {P9_3D[2]:.3f} m"
                    cv2.putText(color_image, tcp_text, (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(color_image, "TCP: No depth data (Z=0)", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if P4_3D and P8_3D:
                    dist_3d = math.sqrt((P8_3D[0] - P4_3D[0])**2 + (P8_3D[1] - P4_3D[1])**2 + (P8_3D[2] - P4_3D[2])**2)
                    dist_mm = dist_3d * 1000 
                    
                    cv2.putText(color_image, f"Gripper 3D Distance: {int(dist_mm)} mm", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                    
                    status = "Close" if dist_mm < 20 else "Open"
                    color = (0, 0, 255) if dist_mm < 40 else (0, 255, 0)
                    cv2.putText(color_image, f"Status: {status}", (20, 130), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(color_image, "[Warning] Cannot find hand", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        depth_colormap_rgb = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
        depth_colormap = cv2.cvtColor(depth_colormap_rgb, cv2.COLOR_RGB2BGR)
        
        if depth_colormap.shape[1] != color_image.shape[1]:
            scale = color_image.shape[1] / depth_colormap.shape[1]
            depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], int(depth_colormap.shape[0] * scale)))
            
        combined_view = np.vstack((color_image, depth_colormap))
        cv2.imshow("Vision Teleoperation Pipeline", combined_view)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    hand_tracker.close()

# ==========================================================
# Cleanup
# ==========================================
def shutdown_pipeline(pipeline):
    try:
        pipeline.stop()
    except:
        pass
    cv2.destroyAllWindows()
    print("[INFO] Pipeline stopped")


# ==========================================================
# Main
# ==========================================
def main():
    USE_PLAYBACK = True
    pipeline = None

    try:
        if USE_PLAYBACK:
            file_path = r"20260706_151013.db3" 
            pipeline, profile = create_playback_pipeline(file_path)
        else:
            pipeline, profile = create_real_camera_pipeline()

        run_vision_pipeline(pipeline, profile)

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if pipeline is not None:
            shutdown_pipeline(pipeline)

if __name__ == "__main__":
    main()