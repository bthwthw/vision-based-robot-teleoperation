import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

class HandTrackerNode:
    def __init__(self, model_path='gesture_recognizer.task'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[TRACKER ERROR] Cannot find '{model_path}'.")
            # https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer

        self.latest_result = None
        self._snap = None
        self.last_mp_timestamp = 0

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            result_callback=self._internal_callback
        )
        self.detector = vision.GestureRecognizer.create_from_options(options)

        self.CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),                 # Ngón cái
            (0, 5), (5, 6), (6, 7), (7, 8),                 # Ngón trỏ
            (5, 9), (9, 10), (10, 11), (11, 12),            # Ngón giữa
            (9, 13), (13, 14), (14, 15), (15, 16),          # Ngón áp út
            (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) # Ngón út
        ]

    def _internal_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def recognize_async(self, bgr_image, timestamp_ms):
        if timestamp_ms <= self.last_mp_timestamp:
            timestamp_ms = self.last_mp_timestamp + 1
        self.last_mp_timestamp = timestamp_ms

        color_image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image_rgb)
        self.detector.recognize_async(mp_image, timestamp_ms)

        # one snapshot per frame so landmarks and gesture come from the same result
        self._snap = self.latest_result

    def draw_skeleton(self, bgr_image):
        res = self._snap
        if not res or not res.hand_landmarks:
            return bgr_image

        h, w, _ = bgr_image.shape
        landmarks = res.hand_landmarks[0]
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
        res = self._snap
        if not res or not res.hand_landmarks:
            return None
        h, w, _ = bgr_image.shape
        lm_list = res.hand_landmarks[0]
        return [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]

    def get_gesture(self):
        """
        return (category_name, score) of the top gesture, or (None, 0.0) if no hand / no gesture
        """
        res = self._snap
        if not res or not res.hand_landmarks or not res.gestures or not res.gestures[0]:
            return None, 0.0
        top = res.gestures[0][0]
        return top.category_name, float(top.score)

    def close(self):
        self.detector.close()