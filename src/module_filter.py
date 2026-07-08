import numpy as np
 
class OneEuroFilter:
    """
    speed (m/s) adaptive One Euro Filter for filtering 3D position data along each axis (x, y, z).

    Parameters:
        min_cutoff: cutoff frequency when speed is low, small -> smoother when stationary, but more lag
        beta: coefficient for increasing cutoff with speed, big -> less lag when moving fast, but more noise
        d_cutoff: cutoff for the derivative (speed), usually default 1.0 is sufficient.
    """
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
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
 
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
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
    def __init__(self, min_cutoff=1.0, beta=0.02):
        self.filters = [OneEuroFilter(min_cutoff, beta) for _ in range(3)]
 
    def filter(self, point_3d, timestamp_s):
        if point_3d is None:
            return None
        return tuple(
            f.filter(point_3d[i], timestamp_s) for i, f in enumerate(self.filters)
        )
 
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
    """
    def __init__(self, min_cutoff=1.5, beta=1.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
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
 
        omega = self._angle_between(self.q_prev, q) / dt
 
        a_d = self._alpha(self.d_cutoff, dt)
        omega_hat = a_d * omega + (1 - a_d) * self.omega_prev
 
        cutoff = self.min_cutoff + self.beta * omega_hat
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