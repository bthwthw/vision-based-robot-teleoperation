import pybullet as p
import random

class SceneManager:
    def __init__(self, table_top_z=0.4):
        self.table_id = None
        self.box_ids = []
        self.table_top_z = table_top_z

    def setup_pick_and_place_scene(self):
        print(f"[Scene INFO] INITIALIZING... Table height: {self.table_top_z} m")

        table_half_extents = [0.3, 0.5, 0.05] 
        table_col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
        table_vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.6, 0.5, 0.4, 1])
        
        z_center_table = self.table_top_z - table_half_extents[2]
        
        self.table_id = p.createMultiBody(
            baseMass=0, 
            baseCollisionShapeIndex=table_col_id, 
            baseVisualShapeIndex=table_vis_id, 
            basePosition=[0.45, 0.0, z_center_table]
        )
        
        p.changeDynamics(self.table_id, -1, lateralFriction=1.0)

        colors = [[0.8, 0.2, 0.2, 1], [0.2, 0.8, 0.2, 1], [0.2, 0.2, 0.8, 1]]
        
        for i in range(3):
            s = random.uniform(0.015, 0.025) 
            box_col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s, s, s])
            box_vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[s, s, s], rgbaColor=colors[i])
            
            x_pos = random.uniform(0.35, 0.55)
            y_pos = random.uniform(-0.4, -0.15)
            
            z_pos = self.table_top_z + s + 0.1 
            
            box_id = p.createMultiBody(
                baseMass=0.1, 
                baseCollisionShapeIndex=box_col_id, 
                baseVisualShapeIndex=box_vis_id, 
                basePosition=[x_pos, y_pos, z_pos]
            )
            
            p.changeDynamics(box_id, -1, lateralFriction=1.5, spinningFriction=0.005, rollingFriction=0.005)
            self.box_ids.append(box_id)
            
        print(f"[Scene INFO] Import {len(self.box_ids)} boxes")