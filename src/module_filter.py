import numpy as np
from collections import deque


class OutlierRejector:
    """
    max_jump_per_sec: (m/s) max speed threshold to reject 1-frame outlier
        - Với position (m/s): tay người di chuyển nhanh hiếm khi vượt 2-3 m/s.
        - Với góc (rad/s): cổ tay xoay nhanh hiếm khi vượt ~15 rad/s (~860 deg/s).
    window: số mẫu lịch sử dùng để tính trung vị tham chiếu.
    """
    def __init__(self, max_jump_per_sec, window=5):
        self.max_jump_per_sec = max_jump_per_sec
        self.history = deque(maxlen=window)
        self.t_prev = None

    def check(self, value, timestamp_s):
        """
        value: scalar (for each axis) or for Quaternion
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
            # Ngoại suy tuyến tính từ 2 điểm gần nhất thay vì nhận giá trị nhiễu
            if len(self.history) >= 2:
                extrapolated = self.history[-1] + (self.history[-1] - self.history[-2])
            else:
                extrapolated = self.history[-1]
            self.history.append(extrapolated)
            self.t_prev = timestamp_s
            return extrapolated, True

        self.history.append(value)
        self.t_prev = timestamp_s
        return value, False

    def reset(self):
        self.history.clear()
        self.t_prev = None


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
        self.dx_prev = 0.0
        self.t_prev = None
 
    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
 
    def filter(self, x, timestamp_s):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = timestamp_s
            return x
 
        dt = timestamp_s - self.t_prev
        if dt <= 0:
            return self.x_prev
 
        # Speed 
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
 
        # Adaptive cutoff, clamp để tránh outlier "mở toang" filter
        cutoff = min(self.min_cutoff + self.beta * abs(dx_hat), self.cutoff_max)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
 
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp_s
        return x_hat
 
    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
 
 
class Position3DFilter:
    """
    reject_max_jump_mps: (m/s) max speed threshold to reject 1-frame outlier - None to disable
    """
    def __init__(self, min_cutoff=1.0, beta=0.02, cutoff_max=15.0, reject_max_jump_mps=2.5):
        self.filters = [OneEuroFilter(min_cutoff, beta, cutoff_max=cutoff_max) for _ in range(3)]
        self.rejectors = (
            [OutlierRejector(reject_max_jump_mps) for _ in range(3)]
            if reject_max_jump_mps is not None else None
        )

    def filter(self, point_3d, timestamp_s):
        if point_3d is None:
            return None

        if self.rejectors is not None:
            point_3d = tuple(
                self.rejectors[i].check(point_3d[i], timestamp_s)[0] for i in range(3)
            )

        return tuple(
            f.filter(point_3d[i], timestamp_s) for i, f in enumerate(self.filters)
        )
 
    def reset(self):
        for f in self.filters:
            f.reset()
        if self.rejectors is not None:
            for r in self.rejectors:
                r.reset()

            
 
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
    def __init__(self, min_cutoff=1.5, beta=1.0, d_cutoff=1.0, cutoff_max=20.0, reject_max_omega=15.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.cutoff_max = cutoff_max
        self.reject_max_omega = reject_max_omega
        self.q_prev = None    
        self.omega_prev = 0.0   
        self.t_prev = None
 
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
            return q
 
        dt = timestamp_s - self.t_prev
        if dt <= 0:
            return self.q_prev

        omega_raw = self._angle_between(self.q_prev, q) / dt

        if self.reject_max_omega is not None and omega_raw > self.reject_max_omega:
            self.t_prev = timestamp_s
            return self.q_prev
 
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