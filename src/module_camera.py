import pyrealsense2 as rs
import numpy as np
import os
import cv2

class RealSenseNode:
    def __init__(self, playback_file=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if playback_file:
            if not os.path.exists(playback_file):
                raise FileNotFoundError(f"[ERROR] Cannot find {playback_file}")
            self.config.enable_device_from_file(playback_file, repeat_playback=True)
            self.profile = self.pipeline.start(self.config)
            self.profile.get_device().as_playback().set_real_time(True)
            print("[INFO] Camera Node started in PLAYBACK mode.")
        else:
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            self.profile = self.pipeline.start(self.config)
            print("[INFO] Camera Node started in REAL-STREAM mode.")

        self.aligner = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()
        
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # Post-processing
        self.depth_to_disparity = rs.disparity_transform(True)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        self.disparity_to_depth = rs.disparity_transform(False)

        self.last_hw_timestamp = 0
        self.timestamp_offset = 0

    def get_frames(self):
        success, frames = self.pipeline.try_wait_for_frames(timeout_ms=5000)
        if not success:
            return None, None, None, None

        aligned_frames = self.aligner.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None, None

        filtered_depth = depth_frame
        filtered_depth = self.depth_to_disparity.process(filtered_depth)
        filtered_depth = self.spatial.process(filtered_depth)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = self.disparity_to_depth.process(filtered_depth).as_depth_frame()

        depth_array = np.asanyarray(filtered_depth.get_data())

        color_image_raw = np.asanyarray(color_frame.get_data())
        fmt = color_frame.profile.format()
        if fmt == rs.format.rgb8:
            import cv2
            color_image = cv2.cvtColor(color_image_raw, cv2.COLOR_RGB2BGR)
        elif fmt == rs.format.rgba8:
            import cv2
            color_image = cv2.cvtColor(color_image_raw, cv2.COLOR_RGBA2BGR)
        else:
            color_image = np.copy(color_image_raw)

        current_hw_timestamp = int(color_frame.get_timestamp())
        if current_hw_timestamp < self.last_hw_timestamp:
            self.timestamp_offset += self.last_hw_timestamp
        self.last_hw_timestamp = current_hw_timestamp
        linear_timestamp = current_hw_timestamp + self.timestamp_offset

        return color_image, filtered_depth, depth_array, linear_timestamp

    def colorize_depth(self, depth_frame):
        depth_colormap_rgb = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return cv2.cvtColor(depth_colormap_rgb, cv2.COLOR_RGB2BGR)

    def extract_3d_coordinates(self, u, v, depth_frame, depth_array, radius=20):
        h, w = depth_array.shape
        x_int, y_int = int(u), int(v)
        
        if not (0 <= x_int < w and 0 <= y_int < h):
            return None
            
        if depth_array[y_int, x_int] > 0:
            best_u, best_v = x_int, y_int
        else:
            y_min, y_max = max(0, y_int - radius), min(h, y_int + radius + 1)
            x_min, x_max = max(0, x_int - radius), min(w, x_int + radius + 1)
            depth_region = depth_array[y_min:y_max, x_min:x_max]
            valid_y_rel, valid_x_rel = np.where(depth_region > 0)
            
            if len(valid_y_rel) == 0:
                return None
            
            valid_x_abs = x_min + valid_x_rel
            valid_y_abs = y_min + valid_y_rel
            distances_sq = (valid_x_abs - u)**2 + (valid_y_abs - v)**2
            radius_sq = radius ** 2
            in_radius_mask = distances_sq <= radius_sq
            
            if not np.any(in_radius_mask):
                return None
                
            min_idx_in_mask = np.argmin(distances_sq[in_radius_mask])
            final_idx = np.where(in_radius_mask)[0][min_idx_in_mask]
            best_u, best_v = int(valid_x_abs[final_idx]), int(valid_y_abs[final_idx])

        depth_value = depth_frame.get_distance(best_u, best_v)
        if depth_value <= 0.0:
            return None
            
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [best_u, best_v], depth_value)

    def stop(self):
        self.pipeline.stop()