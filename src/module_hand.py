import numpy as np
from scipy.spatial.transform import Rotation as R

class HandKinematics:
    """
    3d point to Quaternion and Euler.
    """
    @staticmethod
    def compute_orientation(P0, P5, P9, P17):
        """
        Input: tuple (X, Y, Z) of 4 ponits
        Output: dict include Rotation Matrix, Quaternion (w, x, y, z) and Euler RPY (deg)
        """
        p1 = np.array(P0)
        p2 = np.array(P5)
        p3 = np.array(P9)
        p4 = np.array(P17)

        # Z
        v_5_1 = p2 - p1
        v_17_1 = p4 - p1
        z_axis = np.cross(v_5_1, v_17_1)
        
        # 0?
        norm_z = np.linalg.norm(z_axis)
        if norm_z < 1e-6:
            return None
        z_axis = z_axis / norm_z

        # Y
        v_up = p3 - p1
        y_axis = np.cross(z_axis, v_up)
        
        norm_y = np.linalg.norm(y_axis)
        if norm_y < 1e-6:
            return None
        y_axis = y_axis / norm_y

        # X
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Rotation matrix R (3x3) 
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

        try:
            r = R.from_matrix(rot_matrix)
            
            # Scalar-first (w,x,y,z)
            quat_xyzw = r.as_quat() 
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            euler_rpy = r.as_euler('xyz', degrees=True)
            
            return {
                'matrix': rot_matrix,
                'quaternion': quat_wxyz,
                'rpy': euler_rpy
            }
            
        except Exception as e:
            print(f"[KINEMATICS WARNING] Rotation math error: {e}")
            return None