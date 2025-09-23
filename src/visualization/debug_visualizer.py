# -*- coding: utf-8 -*-
"""
Debug Visualization Manager - Uniformly manages visual elements like bounding boxes, rays, and IK targets.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class DebugVisualizer:
    """
    Unified debug visualization manager.
    Responsible for coordinating the display/hiding of bounding boxes, rays, and the IK target sphere.
    """
    def __init__(self, draw_interface, bbox_visualizer, pickup_assessor, ray_visualizer):
        """
        Initializes the debug visualization manager.

        Args:
            draw_interface: An instance of Isaac Sim's _debug_draw interface.
            bbox_visualizer: The bounding box visualizer.
            pickup_assessor: The pickup assessor.
            ray_visualizer: The ray visualizer.
        """
        self.draw = draw_interface
        self.bbox_visualizer = bbox_visualizer
        self.pickup_assessor = pickup_assessor
        self.ray_visualizer = ray_visualizer
        
        # Master switch controlled by the 'V' key
        self.is_enabled = True
        
        # Reference to the IK target sphere (to be set externally)
        self.ik_target_sphere = None
        
        logger.info("🎨 Debug visualization manager initialized.")

    def toggle_visibility(self):
        """This method is called by the 'V' key to toggle visualization display."""
        self.is_enabled = not self.is_enabled
        
        # Control the display/hiding of the IK target sphere
        self._toggle_ik_target_sphere()
        
        status = "enabled" if self.is_enabled else "disabled"
        print(f"🎨 Debug visualization {status}")
        logger.info(f"🎨 Debug visualization {status}")

    def set_ik_target_sphere(self, target_sphere):
        """Sets the reference to the IK target sphere."""
        self.ik_target_sphere = target_sphere
        logger.info("🎯 IK target sphere reference has been set.")

    def _toggle_ik_target_sphere(self):
        """Toggles the visibility of the IK target sphere."""
        if self.ik_target_sphere is None:
            return
        
        try:
            import omni.usd
            from pxr import UsdGeom
            
            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                # Try multiple possible paths
                target_paths = ["/World/TargetCube", "/World/target", "/World/target_sphere"]
                target_prim = None
                
                for path in target_paths:
                    target_prim = stage.GetPrimAtPath(path)
                    if target_prim.IsValid():
                        break
                
                if target_prim and target_prim.IsValid():
                    imageable = UsdGeom.Imageable(target_prim)
                    if imageable:
                        if self.is_enabled:
                            imageable.MakeVisible()
                            logger.debug("🎯 IK target sphere is now visible.")
                        else:
                            imageable.MakeInvisible()
                            logger.debug("🎯 IK target sphere is now hidden.")
                    else:
                        logger.warning("⚠️ Could not get Imageable interface for target_prim.")
                else:
                    logger.warning(f"⚠️ Could not find the target Prim. Attempted paths: {target_paths}")
            else:
                logger.warning("⚠️ Could not get the USD stage.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to toggle IK target sphere visibility: {e}")

    def update_calculations(self, scene, ik_data, target_configs, step_count):
        """
        Only updates mathematical calculations like collision detection, without performing any drawing.
        This method should be called every frame, regardless of whether visualization is enabled.
        """
        try:
            # Unpack IK data
            ee_pos, ee_rot, gripper_pos, gripper_rot = ik_data
            
            # 1. Calculate ray data
            self.rays_info = self.ray_visualizer.calculate_rays(ee_pos, ee_rot, gripper_pos, gripper_rot)
            
            # 2. Update grasp assessment (ray collision detection)
            self.pickup_assessor.update_and_assess(self.rays_info, target_configs, step_count)
            
            # 3. Update bounding box colors (based on collision detection results)
            for prim_path in target_configs.keys():
                target_configs[prim_path]["obb_color"] = self.pickup_assessor.get_color_for_prim(prim_path)
        
        except Exception as e:
            logger.error(f"❌ Debug visualization calculation failed: {e}")
            self.rays_info = {}

    def draw_visualizations(self, scene, target_configs, step_count):
        """
        Only performs drawing operations and ensures the screen is cleared when disabled.
        """
        # Always clear the drawings from the previous frame
        self.draw.clear_lines()
        
        # If visualization is disabled, return immediately after clearing
        if not self.is_enabled:
            return

        try:
            # 1. Get bounding box drawing data
            bbox_starts, bbox_ends, bbox_colors, bbox_sizes = self.bbox_visualizer.get_lines_for_drawing(
                scene, target_configs, step_count, self.is_enabled
            )
            
            # 2. Get ray drawing data (using cached ray info)
            if hasattr(self, 'rays_info'):
                ray_starts, ray_ends, ray_colors, ray_sizes = self.ray_visualizer.get_rays_for_drawing(
                    self.rays_info, self.is_enabled
                )
            else:
                ray_starts, ray_ends, ray_colors, ray_sizes = [], [], [], []

            # 3. Consolidate all drawing data
            all_starts = bbox_starts + ray_starts
            all_ends = bbox_ends + ray_ends
            all_colors = bbox_colors + ray_colors
            all_sizes = bbox_sizes + ray_sizes
            
            # 4. Draw everything at once
            if all_starts:
                self.draw.draw_lines(all_starts, all_ends, all_colors, all_sizes)
        
        except Exception as e:
            logger.error(f"❌ Debug visualization drawing failed: {e}")

    def update_and_draw(self, scene, ik_data, target_configs, step_count):
        """
        Unified update and draw method - maintained for compatibility, but now calls new methods internally.
        
        Args:
            scene: The Isaac Lab Scene object.
            ik_data: IK calculation result data (ee_pos, ee_rot, gripper_pos, gripper_rot).
            target_configs: Dictionary of target object configurations.
            step_count: The current simulation step.
        """
        # Always perform calculations
        self.update_calculations(scene, ik_data, target_configs, step_count)
        
        # Conditionally perform drawing
        if self.is_enabled:
            self.draw_visualizations(scene, target_configs, step_count)
            
        # Return ray info for use by the state machine
        return getattr(self, 'rays_info', {})
            
    def get_target_state(self, prim_path):
        """
        Gets the state information for a target object.
        
        Args:
            prim_path: The USD path of the object.
            
        Returns:
            dict: A dictionary containing color state information.
        """
        return {
            "is_red": self.pickup_assessor.is_red_state(prim_path),
            "is_pink": self.pickup_assessor.is_pink_state(prim_path),
            "is_green": self.pickup_assessor.is_green_state(prim_path),
            "color": self.pickup_assessor.get_color_for_prim(prim_path)
        }

