import numpy as np

class OutlierRejector:
    """
    Bộ lọc chặn nhiễu tối giản: Cắt bỏ tín hiệu vô lý dựa trên khoảng cách vật lý tối đa.
    """
    def __init__(self, max_jump_per_sec, max_rejects=10):
        self.max_jump_per_sec = max_jump_per_sec
        self.max_rejects = max_rejects
        self.val_logical = None
        self.t_prev = None
        self.reject_count = 0

    def check(self, value, timestamp_s):
        if self.val_logical is None or self.t_prev is None:
            self.val_logical = value
            self.t_prev = timestamp_s
            return value

        dt = timestamp_s - self.t_prev
        self.t_prev = timestamp_s
        if dt <= 0: return self.val_logical

        if abs(value - self.val_logical) > self.max_jump_per_sec * dt:
            self.reject_count += 1
            if self.reject_count >= self.max_rejects:
                self.val_logical = value
                self.reject_count = 0
        else:
            self.val_logical = value
            self.reject_count = 0

        return self.val_logical

    def reset(self):
        self.val_logical = None
        self.t_prev = None
        self.reject_count = 0


class OneEuroFilter:
    """ Bộ lọc thông thấp bậc một 1-Euro. """
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0, cutoff_max=15.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.cutoff_max = cutoff_max
        self.x_prev = None
        self.raw_x_prev = None 
        self.dx_prev = 0.0
 
    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
 
    def filter(self, x, dt):
        if self.x_prev is None:
            self.x_prev = x
            self.raw_x_prev = x
            return x
 
        dx = (x - self.raw_x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
 
        cutoff = min(self.min_cutoff + self.beta * abs(dx_hat), self.cutoff_max)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
 
        self.x_prev = x_hat
        self.raw_x_prev = x
        self.dx_prev = dx_hat
        return x_hat
 
    def reset(self):
        self.x_prev = None
        self.raw_x_prev = None
        self.dx_prev = 0.0


class Scalar1DFilter:
    """ Khối xử lý tịnh tiến: Cắt nhiễu -> Bão hòa -> Làm mượt. """
    def __init__(self, min_cutoff=1.0, beta=0.01, cutoff_max=12.0, reject_max_jump=None, max_rejects=10, slew_limit_mps=5.0):
        self.filter_1e = OneEuroFilter(min_cutoff, beta, cutoff_max=cutoff_max)
        self.rejector = OutlierRejector(reject_max_jump, max_rejects) if reject_max_jump else None
        self.slew_limit_mps = slew_limit_mps
        self.val_physical = None
        self.t_prev = None

    def filter(self, value, timestamp_s):
        if value is None: return None
            
        if self.t_prev is None:
            self.t_prev = timestamp_s
            self.val_physical = value
            if self.rejector: self.rejector.check(value, timestamp_s)
            return self.filter_1e.filter(value, 1.0)

        dt = timestamp_s - self.t_prev
        self.t_prev = timestamp_s
        if dt <= 0: return self.val_physical

        target_val = self.rejector.check(value, timestamp_s) if self.rejector else value
        
        if self.slew_limit_mps:
            max_step = self.slew_limit_mps * dt
            dist = target_val - self.val_physical
            if abs(dist) > max_step:
                target_val = self.val_physical + np.sign(dist) * max_step
        
        self.val_physical = self.filter_1e.filter(target_val, dt)
        return self.val_physical

    def reset(self):
        self.filter_1e.reset()
        if self.rejector: self.rejector.reset()
        self.val_physical = None
        self.t_prev = None


class Position3DFilter:
    def __init__(self, min_cutoff=1.0, beta=0.02, cutoff_max=15.0, reject_max_jump_mps=2.5, max_rejects=10, slew_limit_mps=5.0):
        self.filters = [Scalar1DFilter(min_cutoff, beta, cutoff_max, reject_max_jump_mps, max_rejects, slew_limit_mps) for _ in range(3)]

    def filter(self, point_3d, timestamp_s):
        if point_3d is None: return None
        return tuple(f.filter(point_3d[i], timestamp_s) for i, f in enumerate(self.filters))
 
    def reset(self):
        for f in self.filters: f.reset()


class QuaternionFilter:
    """ Khối xử lý góc xoay tối giản. """
    def __init__(self, min_cutoff=1.5, beta=1.0, d_cutoff=1.0, cutoff_max=20.0, reject_max_omega=15.0, max_rejects=20, slew_limit_omega=10.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.cutoff_max = cutoff_max
        self.reject_max_omega = reject_max_omega
        self.max_rejects = max_rejects
        self.slew_limit_omega = slew_limit_omega
        
        self.q_logical = None    
        self.q_physical = None   
        self.t_prev = None       
        self.reject_count = 0
        self.omega_prev = 0.0   
 
    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / dt)
 
    @staticmethod
    def _angle_between(q0, q1):
        dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
        if dot < 0.0:
            dot = -dot
        return 2.0 * np.arccos(dot)
 
    @staticmethod
    def _slerp(q0, q1, t):
        dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        return (np.cos(theta) - dot * sin_theta / sin_theta_0) * q0 + (sin_theta / sin_theta_0) * q1
 
    def filter(self, quat_wxyz, timestamp_s):
        q = np.array(quat_wxyz, dtype=float)
        q /= np.linalg.norm(q)
 
        if self.t_prev is None:
            self.q_logical = self.q_physical = q
            self.t_prev = timestamp_s
            return q
 
        dt = timestamp_s - self.t_prev
        self.t_prev = timestamp_s
        if dt <= 0: return self.q_physical

        if self.reject_max_omega:
            if self._angle_between(self.q_logical, q) > self.reject_max_omega * dt:
                self.reject_count += 1
                if self.reject_count >= self.max_rejects:
                    self.q_logical = q
                    self.reject_count = 0
            else:
                self.q_logical = q
                self.reject_count = 0
        else:
            self.q_logical = q

        dist_phys_to_log = self._angle_between(self.q_physical, self.q_logical)
        target_q = self.q_logical
        omega_target = dist_phys_to_log / dt if dist_phys_to_log >= 1e-5 else 0.0
        
        if self.slew_limit_omega and omega_target > self.slew_limit_omega:
            safe_t = (self.slew_limit_omega * dt) / dist_phys_to_log
            target_q = self._slerp(self.q_physical, self.q_logical, safe_t)
            omega_target = self.slew_limit_omega

        a_d = self._alpha(self.d_cutoff, dt)
        omega_hat = a_d * omega_target + (1 - a_d) * self.omega_prev
        t = self._alpha(min(self.min_cutoff + self.beta * omega_hat, self.cutoff_max), dt)
 
        self.q_physical = self._slerp(self.q_physical, target_q, t)
        self.omega_prev = omega_hat
        
        return self.q_physical
 
    def reset(self):
        self.q_logical = self.q_physical = self.t_prev = None
        self.omega_prev = self.reject_count = 0