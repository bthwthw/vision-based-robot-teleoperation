import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

class HandTrackerNode:
    def __init__(self, model_path='hand_landmarker.task'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[TRACKER ERROR] Cannot find '{model_path}'.")
            # https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker 

        self.latest_result = None
        self.last_mp_timestamp = 0

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            result_callback=self._internal_callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),                 # Ngón cái
            (0, 5), (5, 6), (6, 7), (7, 8),                 # Ngón trỏ
            (5, 9), (9, 10), (10, 11), (11, 12),            # Ngón giữa
            (9, 13), (13, 14), (14, 15), (15, 16),          # Ngón áp út
            (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) # Ngón út
        ]

    def _internal_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def detect_async(self, bgr_image, timestamp_ms):
        if timestamp_ms <= self.last_mp_timestamp:
            timestamp_ms = self.last_mp_timestamp + 1
        self.last_mp_timestamp = timestamp_ms

        color_image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image_rgb)
        self.detector.detect_async(mp_image, timestamp_ms)

    def draw_skeleton(self, bgr_image):
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return bgr_image

        h, w, _ = bgr_image.shape
        landmarks = self.latest_result.hand_landmarks[0]
        pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for connection in self.CONNECTIONS:
            cv2.line(bgr_image, pixel_pts[connection[0]], pixel_pts[connection[1]], (200, 200, 200), 2)
        for pt in pixel_pts:
            cv2.circle(bgr_image, pt, 4, (0, 215, 255), -1)

        return bgr_image
    
    def get_all_landmarks_pixel(self, bgr_image):
        """
        return List of 21 tuples (u, v) or None if no hand detected
        """
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return None
        h, w, _ = bgr_image.shape
        lm_list = self.latest_result.hand_landmarks[0]
        pixel_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
        
        return pixel_landmarks
    
    def close(self):
        self.detector.close()