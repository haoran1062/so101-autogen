# -*- coding: utf-8 -*-
"""
Multi-Camera Controller - Simplified Version
Uses the Isaac Sim Camera API for camera management and data acquisition.
Reads camera parameters from a configuration file, retaining necessary
position correction and quaternion update logic.
"""

import os
import logging
import numpy as np

# Isaac Sim相关导入
from omni.kit.viewport.utility import get_viewport_from_window_name
from isaacsim.sensors.camera import Camera
import omni.usd
from pxr import Sdf, Gf, UsdGeom

logger = logging.getLogger(__name__)


class MultiCameraController:
    """Multi-Camera Controller - Simplified version that reads parameters from config and retains necessary position correction logic."""
    
    def __init__(self, config=None):
        """
        Initializes the camera controller.
        
        Args:
            config: A scene configuration dictionary containing camera parameters.
        """
        # Get the viewport
        self.viewport = get_viewport_from_window_name("Viewport")
        
        # Camera paths
        self.main_camera_path = "/OmniverseKit_Persp"
        self.wrist_camera_path = "/World/so101_robot/gripper_link/wrist_camera"
        self.front_camera_path = "/World/so101_robot/base_link/front_camera"
        
        # Camera list
        self.cameras = [
            ("Main View", self.main_camera_path),
            ("Wrist Camera", self.wrist_camera_path),
            ("Front Camera", self.front_camera_path)
        ]
        
        self.current_camera_index = 0
        
        # Frame count (for statistics)
        self.frame_count = 0
        
        # Camera objects
        self.wrist_camera = None
        self.front_camera = None
        
        # Read camera parameters from the configuration file
        self._load_camera_config(config)
        
        # Create cameras
        self._create_cameras_with_isaacsim_api()
        
        logger.info("📷 Multi-camera controller initialized (Simplified Version).")
        logger.info("   • Reading camera parameters from config file.")
        logger.info("   • Retaining necessary position correction and quaternion update logic.")
    
    def _load_camera_config(self, config):
        """Loads camera parameters from the configuration file."""
        if config is None:
            # Use default parameters
            self.front_camera_config = {
                "position": [0.52, 0.0, 0.4],
                "orientation": [0.65328, 0.2706, 0.2706, 0.65328],  # (w, x, y, z)
                "focal_length": 28.7,
                "horizontal_aperture": 38.11,
                "vertical_aperture": 28.58,
                "clipping_range": [0.01, 50.0]
            }
            
            self.wrist_camera_config = {
                "position": [0.02, 0.2, 0.1],
                "orientation": [0.93969, -0.34202, 0.0, 0.0],  # (w, x, y, z)
                "focal_length": 36.5,
                "horizontal_aperture": 36.83,
                "vertical_aperture": 27.62,
                "clipping_range": [0.01, 50.0]
            }
        else:
            # Read parameters from the configuration file
            cameras_config = config.get("cameras", {})
            
            # Front camera configuration
            front_config = cameras_config.get("front_camera", {})
            self.front_camera_config = {
                "position": front_config.get("position", [0.52, 0.0, 0.4]),
                "orientation": front_config.get("orientation", [0.65328, 0.2706, 0.2706, 0.65328]),
                "focal_length": front_config.get("focal_length", 28.7),
                "horizontal_aperture": front_config.get("horizontal_aperture", 38.11),
                "vertical_aperture": front_config.get("vertical_aperture", 28.58),
                "clipping_range": front_config.get("clipping_range", [0.01, 50.0])
            }
            
            # Wrist camera configuration
            wrist_config = cameras_config.get("wrist_camera", {})
            self.wrist_camera_config = {
                "position": wrist_config.get("position", [0.02, 0.2, 0.1]),
                "orientation": wrist_config.get("orientation", [0.93969, -0.34202, 0.0, 0.0]),
                "focal_length": wrist_config.get("focal_length", 36.5),
                "horizontal_aperture": wrist_config.get("horizontal_aperture", 36.83),
                "vertical_aperture": wrist_config.get("vertical_aperture", 27.62),
                "clipping_range": wrist_config.get("clipping_range", [0.01, 50.0])
            }
        
        logger.info("📋 Camera configuration loaded:")
        logger.info(f"   Front Camera: Position {self.front_camera_config['position']}, Orientation {self.front_camera_config['orientation']}")
        logger.info(f"   Wrist Camera: Position {self.wrist_camera_config['position']}, Orientation {self.wrist_camera_config['orientation']}")
    
    def update_frame_count(self):
        """Updates the frame count."""
        self.frame_count += 1
    
    def _create_cameras_with_isaacsim_api(self):
        """Creates standard USD Camera Prims using UsdGeom.Camera.Define() and then wraps them with isaacsim.sensors.camera.Camera."""
        
        try:
            logger.info("🎥 Starting creation of standard USD Camera Prims + Isaac Sim Camera wrappers...")
            
            # Create wrist camera
            self._create_standard_camera("Wrist", self.wrist_camera_path, self.wrist_camera_config)
            
            # Ensure the position is correctly applied to the USD transform
            logger.info("🔧 Ensuring position is correctly applied to USD transform...")
            self.fix_wrist_camera_to_desired_position()
            logger.info("✅ Position correctly applied to USD transform.")
            
            # Create front camera
            self._create_standard_camera("Front", self.front_camera_path, self.front_camera_config)
            
            logger.info("✅ Standard USD Camera Prims + Isaac Sim Camera wrappers created successfully.")
            
            # Ensure cameras use the correct quaternion
            logger.info("🔧 Applying correct camera quaternions...")
            self._update_wrist_camera_quaternion(self.wrist_camera_config["orientation"])
            self._update_front_camera_quaternion(self.front_camera_config["orientation"])
            logger.info("✅ Camera quaternions applied.")
            
            # Verify wrist camera position
            logger.info("🔍 Verifying wrist camera position...")
            stage = omni.usd.get_context().get_stage()
            wrist_camera_prim = stage.GetPrimAtPath(self.wrist_camera_path)
            if wrist_camera_prim.IsValid():
                wrist_xform = UsdGeom.Xformable(wrist_camera_prim)
                wrist_world_transform = wrist_xform.ComputeLocalToWorldTransform(0)
                actual_wrist_world_position = wrist_world_transform.ExtractTranslation()
                logger.info(f"🔍 Actual wrist camera world position: {actual_wrist_world_position}")
                logger.info(f"🔍 Desired world position: {self.wrist_camera_config['position']}")
                
                # Calculate error
                position_error = [
                    abs(actual_wrist_world_position[0] - self.wrist_camera_config['position'][0]),
                    abs(actual_wrist_world_position[1] - self.wrist_camera_config['position'][1]),
                    abs(actual_wrist_world_position[2] - self.wrist_camera_config['position'][2])
                ]
                logger.info(f"🔍 Position error: {position_error}")
            else:
                logger.warning("⚠️ Could not find wrist camera Prim for position verification.")
            
        except Exception as e:
            logger.error(f"❌ Camera creation failed: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise
    
    def _update_front_camera_quaternion(self, quat_components):
        """Updates the quaternion for the front camera."""
        try:
            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.GetPrimAtPath(self.front_camera_path)
            
            if camera_prim.IsValid():
                xform = UsdGeom.Xformable(camera_prim)
                existing_ops = xform.GetOrderedXformOps()
                
                # Find the orientation operation
                orient_op = None
                for op in existing_ops:
                    if "orient" in op.GetOpName():
                        orient_op = op
                        break
                
                if orient_op:
                    # Update quaternion
                    quat = Gf.Quatf(*quat_components)  # (w, x, y, z) format
                    orient_op.Set(quat)
                    logger.debug(f"✅ Front camera quaternion updated: {quat_components}")
                else:
                    logger.warning("⚠️ Could not find orientation operation for the front camera.")
            else:
                logger.warning("⚠️ Could not find front camera Prim.")
                
        except Exception as e:
            logger.error(f"❌ Failed to update front camera quaternion: {e}")
    
    def _update_wrist_camera_quaternion(self, quat_components):
        """Updates the quaternion for the wrist camera."""
        try:
            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.GetPrimAtPath(self.wrist_camera_path)
            
            if camera_prim.IsValid():
                xform = UsdGeom.Xformable(camera_prim)
                existing_ops = xform.GetOrderedXformOps()
                
                # Find the orientation operation
                orient_op = None
                for op in existing_ops:
                    if "orient" in op.GetOpName():
                        orient_op = op
                        break
                
                if orient_op:
                    # Update quaternion
                    quat = Gf.Quatf(*quat_components)  # (w, x, y, z) format
                    orient_op.Set(quat)
                    logger.debug(f"✅ Wrist camera quaternion updated: {quat_components}")
                else:
                    logger.warning("⚠️ Could not find orientation operation for the wrist camera.")
            else:
                logger.warning("⚠️ Could not find wrist camera Prim.")
                
        except Exception as e:
            logger.error(f"❌ Failed to update wrist camera quaternion: {e}")
    
    def update_wrist_camera_position(self, new_position):
        """Updates the position of the wrist camera."""
        try:
            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.GetPrimAtPath(self.wrist_camera_path)
            
            if camera_prim.IsValid():
                xform = UsdGeom.Xformable(camera_prim)
                existing_ops = xform.GetOrderedXformOps()
                
                # Find the translation operation
                translate_op = None
                for op in existing_ops:
                    if "translate" in op.GetOpName():
                        translate_op = op
                        break
                
                if translate_op:
                    # Update position
                    translate_op.Set(Gf.Vec3f(*new_position))
                    logger.info(f"✅ Wrist camera position updated: {new_position}")
                    
                    # Verify updated position
                    wrist_world_transform = xform.ComputeLocalToWorldTransform(0)
                    actual_wrist_world_position = wrist_world_transform.ExtractTranslation()
                    logger.info(f"🔍 Actual world position after update: {actual_wrist_world_position}")
                else:
                    logger.warning("⚠️ Could not find translation operation for the wrist camera.")
            else:
                logger.warning("⚠️ Could not find wrist camera Prim.")
                
        except Exception as e:
            logger.error(f"❌ Failed to update wrist camera position: {e}")
    
    def fix_wrist_camera_to_desired_position(self):
        """Corrects the wrist camera to its desired relative position."""
        try:
            # Desired relative position (read from config)
            desired_relative_position = self.wrist_camera_config["position"]
            
            logger.info(f"🔧 Correcting wrist camera to desired relative position: {desired_relative_position}")
            
            # Set the relative position directly
            self.update_wrist_camera_position(desired_relative_position)
            
            # Verify the updated position
            logger.info(f"🔍 Verifying position after correction...")
            stage = omni.usd.get_context().get_stage()
            wrist_camera_prim = stage.GetPrimAtPath(self.wrist_camera_path)
            
            if wrist_camera_prim.IsValid():
                wrist_xform = UsdGeom.Xformable(wrist_camera_prim)
                wrist_world_transform = wrist_xform.ComputeLocalToWorldTransform(0)
                actual_world_position = wrist_world_transform.ExtractTranslation()
                logger.info(f"🔍 World position after correction: {actual_world_position}")
                
                # Check the position displayed in the GUI (relative position)
                ops = wrist_xform.GetOrderedXformOps()
                for op in ops:
                    if "translate" in op.GetOpName():
                        translate_value = op.Get()
                        logger.info(f"🔍 GUI displayed position (relative): {translate_value}")
                        break
            else:
                logger.warning("⚠️ Could not find wrist camera Prim.")
                
        except Exception as e:
            logger.error(f"❌ Failed to correct wrist camera position: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
    
    def _create_standard_camera(self, name, prim_path, config):
        """Creates a standard USD Camera Prim and wraps it as an Isaac Sim Camera."""
        try:
            logger.info(f"🎥 Creating standard USD Camera Prim for {name}...")
            
            # Get the USD stage
            stage = omni.usd.get_context().get_stage()
            
            # 1. Create a standard USD Camera Prim using UsdGeom.Camera.Define()
            camera_prim = UsdGeom.Camera.Define(stage, prim_path)
            
            # 2. Set the world transform (position and orientation)
            xform = UsdGeom.Xformable(camera_prim)
            
            # Set position
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(*config["position"]))
            
            # Set orientation (quaternion format)
            orient_op = xform.AddOrientOp()
            quat = Gf.Quatf(*config["orientation"])  # (w, x, y, z) format
            orient_op.Set(quat)
            
            # 3. Set core camera parameters (focal length, etc.)
            camera_prim.GetFocalLengthAttr().Set(config["focal_length"])
            camera_prim.GetHorizontalApertureAttr().Set(config["horizontal_aperture"])
            camera_prim.GetVerticalApertureAttr().Set(config["vertical_aperture"])
            camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(config["clipping_range"][0], config["clipping_range"][1]))
            
            # 4. Set camera display properties
            camera_prim_obj = camera_prim.GetPrim()
            
            # Set camera display properties
            if "wrist" in prim_path:
                camera_prim_obj.CreateAttribute("displayColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))  # Red - Wrist Camera
            else:
                camera_prim_obj.CreateAttribute("displayColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))  # Green - Front Camera
            camera_prim_obj.CreateAttribute("displayOpacity", Sdf.ValueTypeNames.Float).Set(0.8)
            camera_prim_obj.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("visible")
            
            # 5. Lock camera position - to prevent modification in the GUI
            camera_prim_obj.SetCustomDataByKey("lock_camera", True)
            camera_prim_obj.SetCustomDataByKey("camera_type", "fixed_sensor")
            camera_prim_obj.SetCustomDataByKey("description", f"Locked {name} camera for data collection")
            
            logger.info(f"🔒 {name} camera position is locked to prevent GUI modification.")
            
            # Ensure the camera has the correct transform order
            xform.SetXformOpOrder([translate_op, orient_op])
            
            logger.info(f"✅ Standard USD Camera Prim for {name} created successfully.")
            logger.debug(f"   Position: {config['position']}")
            logger.debug(f"   Orientation: {config['orientation']}")
            logger.debug(f"   Focal Length: {config['focal_length']}mm")
            
            # 6. Now, wrap this existing Prim into the Isaac Sim Camera class
            logger.info(f"🎥 Wrapping {name} camera as an Isaac Sim Camera sensor...")
            
            if name == "Wrist":
                self.wrist_camera = Camera(
                    prim_path=prim_path,
                    position=np.array(config["position"]),
                    orientation=np.array(config["orientation"]),
                    frequency=30,
                    resolution=(640, 480)
                )
                self.wrist_camera.initialize()
                logger.info("✅ Wrist camera sensor wrapped and initialized successfully.")
            else:
                self.front_camera = Camera(
                    prim_path=prim_path,
                    position=np.array(config["position"]),
                    orientation=np.array(config["orientation"]),
                    frequency=30,
                    resolution=(640, 480)
                )
                self.front_camera.initialize()
                logger.info("✅ Front camera sensor wrapped and initialized successfully.")
            
        except Exception as e:
            logger.error(f"❌ Failed to create {name} camera: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise
    
    def switch_camera(self):
        """Switches the camera view."""
        if self.viewport is None:
            logger.warning("⚠️ Viewport not available, cannot switch cameras.")
            return
            
        self.current_camera_index = (self.current_camera_index + 1) % len(self.cameras)
        camera_name, camera_path = self.cameras[self.current_camera_index]
        
        try:
            # Switch camera directly using the set_active_camera method
            self.viewport.set_active_camera(camera_path)
            print(f"📷 Switched to: {camera_name}")
            logger.info(f"📷 Switched camera to: {camera_name}")
        except Exception as e:
            logger.error(f"❌ Failed to switch camera: {e}")
    
    def get_current_camera_info(self):
        """Gets information about the current camera."""
        if self.current_camera_index < len(self.cameras):
            camera_name, camera_path = self.cameras[self.current_camera_index]
            return {
                "name": camera_name,
                "path": camera_path,
                "index": self.current_camera_index
            }
        return None