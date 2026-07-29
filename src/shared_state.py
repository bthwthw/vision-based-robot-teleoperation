import threading

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

class SharedLogState:
    def __init__(self):
        self._lock = threading.Lock()
        self._vision_log = {}
        self._control_log = {}

    def write_vision(self, data):
        with self._lock:
            self._vision_log = data.copy()

    def write_control(self, data):
        with self._lock:
            self._control_log = data.copy()

    def read_all(self):
        with self._lock:
            return self._vision_log.copy(), self._control_log.copy()