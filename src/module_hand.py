import numpy as np
from scipy.spatial.transform import Rotation as R


class HandKinematics:
    """
    3d point to Quaternion and Euler.
    """
    @staticmethod
    def compute_orientation(P0, Px1, Px2, P3, handedness="Left"):
        """
        Coordinate System: X (Red-Upward), Y (Green), Z (Blue-Approach)
        Input: tuple (X, Y, Z) of 4 ponits
        Output: dict include Rotation Matrix, Quaternion (w, x, y, z) and Euler RPY (deg)
        """
        p0 = np.array(P0)
        p1 = np.array(Px1)
        p2 = np.array(Px2)
        p3 = np.array(P3)

        # X
        x_axis = p2 - p1
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-6:
            return None
        x_axis = x_axis / norm_x

        
        v_2_0 = p2 - p0
        v_3_0 = p3 - p0
        temp_up = np.cross(v_2_0, v_3_0) 

        if handedness == "Right":
            temp_up = -temp_up
            # print ("[KINEMATICS INFO] Using Right-Handed Coordinate System.")

        # Y
        y_axis = np.cross(temp_up, x_axis)
        norm_y = np.linalg.norm(y_axis)
        if norm_y < 1e-6:
            return None
        y_axis = y_axis / norm_y

        # Z
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

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
            
        except ValueError as e:
            print(f"[KINEMATICS WARNING] Rotation math error: {e}")
            return None