import numpy as np
from collections import deque


class OutlierRejector:
    """
    max_jump_per_sec: (m/s) max speed threshold to reject 1-frame outlier
    window: số mẫu lịch sử dùng để tính trung vị tham chiếu
    max_rejects: Số lượng Suspect liên tiếp tối đa trước khi reset
    """
    def __init__(self, max_jump_per_sec, window=5, max_rejects=10):
        self.max_jump_per_sec = max_jump_per_sec
        self.window_size = window
        self.max_rejects = max_rejects
        self.history = deque(maxlen=window)
        self.t_prev = None
        self.reject_count = 0

    def check(self, value, timestamp_s):
        """
        return: (filtered_value, is_suspect_flag) (for each axis) or for Quaternion
        """
        if self.t_prev is None or len(self.history) < 2:
            self.history.append(value)
            self.t_prev = timestamp_s
            return value, False

        dt = timestamp_s - self.t_prev
        if dt <= 0:
            return self.history[-1], False

        median_ref = np.median(self.history)
        delta = abs(value - median_ref)
        limit = self.max_jump_per_sec * dt

        if delta > limit:
            self.reject_count += 1
            if self.reject_count >= self.max_rejects:
                self.history.clear()
                self.history.append(value)
                self.t_prev = timestamp_s
                self.reject_count = 0
                return value, False
            return self.history[-1], True

        self.reject_count = 0
        self.history.append(value)
        self.t_prev = timestamp_s
        return value, False

    def reset(self):
        self.history.clear()
        self.t_prev = None
        self.reject_count = 0


class OneEuroFilter:
    """
    speed (m/s) adaptive One Euro Filter for filtering 3D position data along each axis (x, y, z).

    Parameters:
        min_cutoff: cutoff frequency when speed is low, small -> smoother when stationary, but more lag
        beta: coefficient for increasing cutoff with speed, big -> less lag when moving fast, but more noise
        d_cutoff: cutoff for the derivative (speed), usually default 1.0 is sufficient.
        cutoff_max: (Hz) max cutoff frequency to avoid spike/outlier 
    """
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0, cutoff_max=15.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.cutoff_max = cutoff_max
        self.x_prev = None
        self.raw_x_prev = None 
        self.dx_prev = 0.0
        self.t_prev = None
 
    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
 
    def filter(self, x, timestamp_s):
        if self.t_prev is None:
            self.x_prev = x
            self.raw_x_prev = x
            self.t_prev = timestamp_s
            return x
 
        dt = timestamp_s - self.t_prev
        if dt <= 0:
            return self.x_prev
 
        dx = (x - self.raw_x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
 
        cutoff = min(self.min_cutoff + self.beta * abs(dx_hat), self.cutoff_max)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
 
        self.x_prev = x_hat
        self.raw_x_prev = x
        self.dx_prev = dx_hat
        self.t_prev = timestamp_s
        return x_hat
 
    def reset(self):
        self.x_prev = None
        self.raw_x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
 

class Scalar1DFilter:
    """
    1D filter incorporate: Outlier Rejector and 1 Euro Filter
    """
    def __init__(self, min_cutoff=1.0, beta=0.01, cutoff_max=12.0, reject_max_jump=None, max_rejects=10):
        self.filter_1e = OneEuroFilter(min_cutoff, beta, cutoff_max=cutoff_max)
        self.rejector = OutlierRejector(reject_max_jump, max_rejects=max_rejects) if reject_max_jump is not None else None

    def filter(self, value, timestamp_s):
        if value is None:
            return None
            
        if self.rejector is not None:
            checked_val, is_suspect = self.rejector.check(value, timestamp_s)
            
            if is_suspect:
                if self.filter_1e.x_prev is not None:
                    return self.filter_1e.x_prev
                return checked_val
            
            value = checked_val
            
        return self.filter_1e.filter(value, timestamp_s)

    def reset(self):
        self.filter_1e.reset()
        if self.rejector is not None:
            self.rejector.reset()

 

class Position3DFilter:
    """
    3D filter using 1D filters for each axis (x, y, z).
    reject_max_jump_mps: (m/s) max speed threshold to reject 1-frame outlier - None to disable
    """
    def __init__(self, min_cutoff=1.0, beta=0.02, cutoff_max=15.0, reject_max_jump_mps=2.5, max_rejects=10):
        self.filters = [
            Scalar1DFilter(min_cutoff, beta, cutoff_max, reject_max_jump_mps, max_rejects) 
            for _ in range(3)
        ]

    def filter(self, point_3d, timestamp_s):
        if point_3d is None:
            return None
        return tuple(f.filter(point_3d[i], timestamp_s) for i, f in enumerate(self.filters))
 
    def reset(self):
        for f in self.filters:
            f.reset()
 

class QuaternionFilter:
    """
    speed (rad/s) adaptive SLERP filter for filtering quaternion orientation data (w, x, y, z).
    
    Parameters:
        min_cutoff: cutoff frequency when speed is low, small -> smoother when stationary, but more lag
        beta: coefficient for increasing cutoff with speed, big -> less lag when moving fast, but more noise
        d_cutoff: cutoff for the derivative (speed), usually default 1.0 is sufficient.
        cutoff_max: avoid spike/outlier 
        reject_max_omega: (rad/s) max angular speed threshold to reject 1-frame outlier - None to disable
    """
    def __init__(self, min_cutoff=1.5, beta=1.0, d_cutoff=1.0, cutoff_max=20.0, reject_max_omega=15.0, max_rejects=10):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.cutoff_max = cutoff_max
        self.reject_max_omega = reject_max_omega
        self.max_rejects = max_rejects
        
        self.q_prev = None    
        self.omega_prev = 0.0   
        self.t_prev = None
        self.reject_count = 0
 
    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
 
    @staticmethod
    def _angle_between(q0, q1):
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)
 
    @staticmethod
    def _slerp(q0, q1, t):
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * q0 + s1 * q1
 
    def filter(self, quat_wxyz, timestamp_s):
        q = np.array(quat_wxyz, dtype=float)
        q = q / np.linalg.norm(q)
 
        if self.q_prev is None or self.t_prev is None:
            self.q_prev = q
            self.t_prev = timestamp_s
            self.omega_prev = 0.0
            self.reject_count = 0
            return q
 
        dt = timestamp_s - self.t_prev
        if dt <= 0:
            return self.q_prev

        # Rate of Change Test cho Vận tốc góc
        omega_raw = self._angle_between(self.q_prev, q) / dt

        if self.reject_max_omega is not None and omega_raw > self.reject_max_omega:
            self.reject_count += 1
            if self.reject_count >= self.max_rejects:
                self.q_prev = q
                self.t_prev = timestamp_s
                self.omega_prev = 0.0
                self.reject_count = 0
                return q
            
            return self.q_prev
 
        self.reject_count = 0
        a_d = self._alpha(self.d_cutoff, dt)
        omega_hat = a_d * omega_raw + (1 - a_d) * self.omega_prev
 
        cutoff = min(self.min_cutoff + self.beta * omega_hat, self.cutoff_max)
        t = self._alpha(cutoff, dt)
 
        q_filtered = self._slerp(self.q_prev, q, t)
 
        self.q_prev = q_filtered
        self.omega_prev = omega_hat
        self.t_prev = timestamp_s
        return q_filtered
 
    def reset(self):
        self.q_prev = None
        self.omega_prev = 0.0
        self.t_prev = None
        self.reject_count = 0