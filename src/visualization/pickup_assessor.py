# -*- coding: utf-8 -*-
"""
Pickup Assessor - Ray collision detection and color state determination.
Ported from the reference script interactive_ik_isaaclab_fixed_cameras.py
"""

import numpy as np
import logging
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

class PickupAssessor:
    """
    The core logic for assessing grasp feasibility.
    Completely decouples aiming and distance based on pure mathematical geometry calculations.
    """
    def __init__(self, scene, bbox_visualizer):
        """
        Initializes the Pickup Assessor.

        Args:
            scene: The Isaac Lab Scene object, used to get object instances.
            bbox_visualizer (BoundingBoxVisualizer): An instance of the bounding box visualization tool.
        """
        self._scene = scene
        self._bbox_visualizer = bbox_visualizer
        self.hit_states = {} # Stores the final hit state for each object
        logger.info("🧠 Pickup Assessor initialized (V3 - Pure geometric calculation).")
    
    def _check_red_ray_alignment(self, prim_path: str, prim_name: str, ray_origin: np.ndarray) -> bool:
        """
        Determines if the vertical projection of the red ray falls within the central 1/4 area of the object.
        """
        prim_object = self._scene.scene.get_object(prim_name)
        if prim_object is None: return False
        
        position, quat_w = prim_object.get_world_pose()
        half_extents = self._bbox_visualizer._prim_extents_cache.get(prim_path)
        center_offset = self._bbox_visualizer._prim_center_offset_cache.get(prim_path)

        if half_extents is None or center_offset is None: return False

        rot_w = Rotation.from_quat([quat_w[1], quat_w[2], quat_w[3], quat_w[0]])
        center_offset_world = rot_w.apply(center_offset)
        geom_center_world = position + center_offset_world
        
        dx = ray_origin[0] - geom_center_world[0]
        dy = ray_origin[1] - geom_center_world[1]
        
        center_threshold_x = half_extents[0] / 2.0
        center_threshold_y = half_extents[1] / 2.0
        
        if abs(dx) < center_threshold_x and abs(dy) < center_threshold_y:
            return True
            
        return False

    def _check_ray_obb_intersection(self, prim_path: str, prim_name: str, ray_origin: np.ndarray, ray_dir: np.ndarray) -> bool:
        """
        Determines if an infinitely long ray intersects with the object's OBB.
        Implemented using the Slab Test algorithm.
        """
        prim_object = self._scene.scene.get_object(prim_name)
        if prim_object is None: return False
        
        position, quat_w = prim_object.get_world_pose()
        half_extents = self._bbox_visualizer._prim_extents_cache.get(prim_path)
        center_offset = self._bbox_visualizer._prim_center_offset_cache.get(prim_path)

        if half_extents is None or center_offset is None: return False

        rot_w = Rotation.from_quat([quat_w[1], quat_w[2], quat_w[3], quat_w[0]])
        center_offset_world = rot_w.apply(center_offset)
        obb_center = position + center_offset_world
        
        # The three axes of the OBB in world coordinates
        obb_axes = rot_w.as_matrix().T

        # Transform the ray origin to the OBB's local coordinate system
        delta = ray_origin - obb_center
        
        # Initialize t_min and t_max for the Slab Test
        t_min = 0.0
        t_max = float('inf')

        for i in range(3): # Test against the three axes of the OBB
            axis = obb_axes[i]
            e = np.dot(axis, delta)
            f = np.dot(axis, ray_dir)
            
            # If the ray is parallel to the slab's plane
            if abs(f) > 1e-6:
                t1 = (-e + half_extents[i]) / f
                t2 = (-e - half_extents[i]) / f
                
                if t1 > t2: t1, t2 = t2, t1 # Ensure t1 is the smaller intersection
                
                if t1 > t_min: t_min = t1
                if t2 < t_max: t_max = t2
                
                # If the intervals do not overlap, the ray does not intersect the OBB
                if t_min > t_max:
                    return False
            # If the ray is parallel but not inside the slab, it does not intersect
            elif -e - half_extents[i] > 0 or -e + half_extents[i] < 0:
                return False
        
        return True

    def update_and_assess(self, rays_info: dict, target_prims: dict, step_count: int):
        """
        Updates the state of all target objects based on the latest ray geometry and object poses.
        """
        self.hit_states = {path: {"hit_by": set(), "is_center_hit": False} for path in target_prims.keys()}
        
        for prim_path, config in target_prims.items():
            prim_name = config["name"]
            
            # 1. Red ray: Vertical projection check
            if self._check_red_ray_alignment(prim_path, prim_name, rays_info["red_ray"]["origin"]):
                self.hit_states[prim_path]["hit_by"].add("red_ray")
                self.hit_states[prim_path]["is_center_hit"] = True
            
            # 2. Purple ray: Mathematical OBB intersection check
            if self._check_ray_obb_intersection(prim_path, prim_name, rays_info["purple_ray"]["origin"], rays_info["purple_ray"]["dir"]):
                self.hit_states[prim_path]["hit_by"].add("purple_ray")
                
            # 3. Green ray: Mathematical OBB intersection check
            if self._check_ray_obb_intersection(prim_path, prim_name, rays_info["green_ray"]["origin"], rays_info["green_ray"]["dir"]):
                 self.hit_states[prim_path]["hit_by"].add("green_ray")

    def get_color_for_prim(self, prim_path: str) -> tuple:
        """
        Determines the OBB color for an object based on its final state and priority rules.
        """
        state = self.hit_states.get(prim_path)
        if not state or not state["hit_by"]:
            return (0.0, 1.0, 1.0, 1.0) # Cyan (default)

        hit_by_rays = state["hit_by"]

        # Priority 1 (Highest): Magenta (fixed jaw collision)
        if "purple_ray" in hit_by_rays:
            return (1.0, 0.0, 1.0, 1.0) # Magenta

        # Priority 2: Green (graspable)
        if "green_ray" in hit_by_rays:
            return (0.0, 1.0, 0.0, 1.0) # Green

        # Priority 3: Red (aligned successfully)
        if "red_ray" in hit_by_rays and state["is_center_hit"]:
            return (1.0, 0.0, 0.0, 1.0) # Red
        
        # Priority 4: Yellow (general hit, e.g., red ray not centered)
        if "red_ray" in hit_by_rays:
            return (1.0, 1.0, 0.0, 1.0) # Yellow
        
        return (0.0, 1.0, 1.0, 1.0) # Cyan (default)
    
    def is_red_state(self, prim_path: str) -> bool:
        """Checks if the object is in the red state (aligned successfully)."""
        color = self.get_color_for_prim(prim_path)
        return color == (1.0, 0.0, 0.0, 1.0)  # Red
    
    def is_pink_state(self, prim_path: str) -> bool:
        """Checks if the object is in the pink/magenta state (collision risk)."""
        color = self.get_color_for_prim(prim_path)
        return color == (1.0, 0.0, 1.0, 1.0)  # Magenta
    
    def is_green_state(self, prim_path: str) -> bool:
        """Checks if the object is in the green state (graspable)."""
        color = self.get_color_for_prim(prim_path)
        return color == (0.0, 1.0, 0.0, 1.0)  # Green

