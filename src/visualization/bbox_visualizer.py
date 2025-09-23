# -*- coding: utf-8 -*-
"""
Bounding Box Visualizer
Ported from the reference script interactive_ik_isaaclab_fixed_cameras.py
"""

import numpy as np
import logging
from omni.isaac.core.utils.bounds import create_bbox_cache, compute_combined_aabb
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

class BoundingBoxVisualizer:
    """
    A helper class to compute and draw 3D oriented bounding boxes (OBB) for objects
    by fetching their real-time poses and using pre-computed dimensions.
    It also supports drawing standard axis-aligned bounding boxes (AABB) for comparison.
    """
    def __init__(self, draw_interface):
        """
        Initializes the BoundingBoxVisualizer.

        Args:
            draw_interface: An instance of Isaac Sim's _debug_draw interface.
        """
        self.draw = draw_interface
        self._prim_extents_cache = {}  # Cache for object half-extents {prim_path: half_extents}
        self._prim_center_offset_cache = {} # Cache for local geometric center offset {prim_path: offset}
        self._bbox_cache_for_init = create_bbox_cache() # Used only for initialization
        logger.info("📦 Custom real-time OBB visualizer initialized (supports AABB/OBB).")

    def cache_prim_extents_and_offset(self, scene, prim_path: str, prim_name: str):
        """
        Computes and caches the dimensions and geometric center offset of an object.
        This method should be called once after the object is loaded and before the simulation starts,
        while the object is in its initial pose.

        Args:
            scene: The Isaac Lab Scene object.
            prim_path (str): The USD path of the object.
            prim_name (str): The registered name of the object in the Scene.
        """
        try:
            # 1. Compute the initial AABB to get its bounds and center in world coordinates
            aabb_bounds = compute_combined_aabb(self._bbox_cache_for_init, prim_paths=[prim_path])
            if aabb_bounds is None or not np.all(np.isfinite(aabb_bounds)):
                logger.warning(f"⚠️ Could not compute a valid AABB for {prim_path}, skipping cache.")
                return

            min_c, max_c = aabb_bounds[:3], aabb_bounds[3:]
            half_extents = (max_c - min_c) / 2.0
            aabb_center_world = (min_c + max_c) / 2.0

            # 2. Get the object's origin position in world coordinates
            prim_object = scene.scene.get_object(prim_name)
            if prim_object is None:
                logger.error(f"❌ Could not find object named {prim_name} while caching dimensions.")
                return
            prim_pos_world, _ = prim_object.get_world_pose()  # Isaac Sim API

            # 3. Compute the center offset vector (in the initial state, world offset equals local offset)
            local_center_offset = aabb_center_world - prim_pos_world

            # 4. Cache the dimensions and offset
            self._prim_extents_cache[prim_path] = half_extents
            self._prim_center_offset_cache[prim_path] = local_center_offset
            logger.info(f"✅ Cached {prim_path} -> Extents: {half_extents.tolist()}, Center Offset: {local_center_offset.tolist()}")

        except Exception as e:
            logger.error(f"❌ Error caching dimensions for {prim_path}: {e}")

    def _compute_vertices_from_corners(self, min_corner, max_corner):
        """Computes the 8 vertices of an AABB from its min and max corners."""
        return [
            np.array([min_corner[0], min_corner[1], min_corner[2]]),
            np.array([max_corner[0], min_corner[1], min_corner[2]]),
            np.array([max_corner[0], max_corner[1], min_corner[2]]),
            np.array([min_corner[0], max_corner[1], min_corner[2]]),
            np.array([min_corner[0], min_corner[1], max_corner[2]]),
            np.array([max_corner[0], min_corner[1], max_corner[2]]),
            np.array([max_corner[0], max_corner[1], max_corner[2]]),
            np.array([min_corner[0], max_corner[1], max_corner[2]]),
        ]

    def _compute_local_vertices_from_extents(self, half_extents):
        """Computes the 8 local vertices of an OBB from its half-extents."""
        x, y, z = half_extents
        return [
            np.array([-x, -y, -z]), np.array([x, -y, -z]),
            np.array([x, y, -z]), np.array([-x, y, -z]),
            np.array([-x, -y, z]), np.array([x, -y, z]),
            np.array([x, y, z]), np.array([-x, y, z]),
        ]

    def _get_edges_from_vertices(self, vertices):
        """Generates index pairs for the 12 edges from 8 vertices."""
        return [
            (vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[3]), (vertices[3], vertices[0]),
            (vertices[4], vertices[5]), (vertices[5], vertices[6]), (vertices[6], vertices[7]), (vertices[7], vertices[4]),
            (vertices[0], vertices[4]), (vertices[1], vertices[5]), (vertices[2], vertices[6]), (vertices[3], vertices[7]),
        ]

    def get_lines_for_drawing(self, scene, prim_configs: dict, step_count: int, is_enabled: bool = True):
        """
        Computes line data for AABBs and OBBs for the given objects in real-time.

        Args:
            scene: The Isaac Lab Scene object, used to get object instances.
            prim_configs (dict): A dictionary of objects and their drawing configurations.
            step_count (int): The current simulation step, used to control logging frequency.
            is_enabled (bool): Whether to enable visualization drawing.
        
        Returns:
            Tuple: Lists containing the start points, end points, colors, and sizes for all lines.
        """
        # Check if debug visualization is enabled
        if not is_enabled:
            return [], [], [], []
        
        all_start_points, all_end_points, all_colors, all_sizes = [], [], [], []
        temp_aabb_cache = create_bbox_cache() # AABB is computed in real-time, use a new cache each time

        for prim_path, config in prim_configs.items():
            # --- Compute and add AABB lines (Yellow) ---
            if config.get('draw_aabb', False):
                try:
                    aabb_bounds = compute_combined_aabb(temp_aabb_cache, prim_paths=[prim_path])
                    if aabb_bounds is not None and np.all(np.isfinite(aabb_bounds)):
                        min_c, max_c = aabb_bounds[:3], aabb_bounds[3:]
                        vertices = self._compute_vertices_from_corners(min_c, max_c)
                        edges = self._get_edges_from_vertices(vertices)
                        for start_v, end_v in edges:
                            all_start_points.append(start_v)
                            all_end_points.append(end_v)
                            all_colors.append(config['aabb_color'])
                            all_sizes.append(1.0)
                except Exception as e:
                    if step_count % 300 == 0:
                        logger.warning(f"❌ Error computing AABB for {prim_path}: {e}")

            # --- Compute and add custom OBB lines (Cyan/Blue) ---
            if config.get('draw_obb', False):
                # Check if extents and offset have been cached
                if prim_path not in self._prim_extents_cache or prim_path not in self._prim_center_offset_cache:
                    if step_count % 300 == 0:
                        logger.warning(f"⚠️ Extents or offset for {prim_path} not cached, cannot compute OBB.")
                    continue

                try:
                    # 1. Get the RigidObject instance
                    prim_object = scene.scene.get_object(config["name"])
                    if prim_object is None:
                         if step_count % 300 == 0:
                            logger.warning(f"⚠️ Could not find object named {config['name']}, cannot compute OBB.")
                         continue
                    
                    # 2. Get the real-time pose (Isaac Sim API)
                    position, orientation_quat = prim_object.get_world_pose()  # Position and quaternion
                    
                    # 3. Read extents and offset from cache, and compute local vertices
                    half_extents = self._prim_extents_cache[prim_path]
                    local_center_offset = self._prim_center_offset_cache[prim_path]
                    local_verts_at_origin = self._compute_local_vertices_from_extents(half_extents)
                    
                    # 4. (Key step) Translate local vertices to the geometric center
                    local_verts_centered = [v + local_center_offset for v in local_verts_at_origin]

                    # 5. Perform coordinate transformation: (centered) local -> world
                    # Scipy requires quaternion in (x, y, z, w) format
                    rotation = Rotation.from_quat([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]])
                    world_verts = [position + rotation.apply(v) for v in local_verts_centered]
                    
                    # 6. Generate edges and add to the drawing list
                    edges = self._get_edges_from_vertices(world_verts)
                    for start_v, end_v in edges:
                        all_start_points.append(start_v)
                        all_end_points.append(end_v)
                        all_colors.append(config['obb_color'])
                        all_sizes.append(1.5)
                except Exception as e:
                    if step_count % 300 == 0:
                        logger.error(f"❌ Error computing custom OBB for {prim_path}: {e}")

        return all_start_points, all_end_points, all_colors, all_sizes
