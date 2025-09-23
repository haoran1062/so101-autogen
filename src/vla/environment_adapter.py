# -*- coding: utf-8 -*-
"""
Isaac Sim Environment Adapter
Adapts our Isaac Sim environment to match the data format expected by the VLA model.
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class IsaacSimEnvironmentAdapter:
    """
    Isaac Sim Environment Adapter
    Responsible for data format conversion and environment interaction.
    """
    
    def __init__(self, world, robot, camera_controller):
        """
        Initializes the environment adapter.
        
        Args:
            world: The Isaac Sim World object.
            robot: The robot object.
            camera_controller: The camera controller.
        """
        self.world = world
        self.robot = robot
        self.camera_controller = camera_controller
        
        logger.info("✅ Isaac Sim Environment Adapter initialized.")
    
    def get_observation(self, task_description: str = "pick and place the orange"):
        """
        Gets observation data in a format compatible with LeRobot.
        
        Args:
            task_description: The description of the task.
            
        Returns:
            dict: A dictionary of observation data.
                - front: Front camera image (480, 640, 3) uint8
                - wrist: Wrist camera image (480, 640, 3) uint8
                - joint_pos: Joint positions (6,) float32
                - task_description: The task description string.
        """
        try:
            # Get camera images
            front_image = self._get_camera_image("front")
            wrist_image = self._get_camera_image("wrist")
            
            # Get joint positions
            joint_pos = self.robot.get_joint_positions()[:6]  # Use only the first 6 joints
            
            # Construct the observation data, following the reference implementation format.
            # The reference implementation expects torch.Tensor format, and camera images need a batch dimension.
            observation = {
                "front": torch.from_numpy(front_image[None, ...]),  # (1, H, W, C)
                "wrist": torch.from_numpy(wrist_image[None, ...]),  # (1, H, W, C)
                "joint_pos": torch.from_numpy(joint_pos[None, ...]),  # (1, 6)
                "task_description": task_description
            }
            
            return observation
            
        except Exception as e:
            logger.error(f"❌ Failed to get observation data: {e}")
            # Return default observation data on failure
            return self._get_default_observation(task_description)
    
    def _get_camera_image(self, camera_name: str):
        """
        Gets image data from the specified camera.
        
        Args:
            camera_name: The name of the camera ("front" or "wrist").
            
        Returns:
            np.ndarray: Image data (480, 640, 3) uint8.
        """
        try:
            if camera_name == "front" and hasattr(self.camera_controller, 'front_camera'):
                camera = self.camera_controller.front_camera
            elif camera_name == "wrist" and hasattr(self.camera_controller, 'wrist_camera'):
                camera = self.camera_controller.wrist_camera
            else:
                logger.warning(f"⚠️ Camera '{camera_name}' is not available.")
                return self._get_default_image()
            
            # Get RGBA image
            rgba_data = camera.get_rgba()
            
            if rgba_data is None:
                logger.warning(f"⚠️ Could not get image from '{camera_name}' camera.")
                return self._get_default_image()
            
            # Convert to RGB format
            if rgba_data.shape == (480, 640, 4):
                # RGBA -> RGB
                rgb_data = rgba_data[:, :, :3]
            elif rgba_data.shape == (480, 640, 3):
                # Already in RGB format
                rgb_data = rgba_data
            else:
                logger.warning(f"⚠️ Unexpected image format from '{camera_name}' camera: {rgba_data.shape}")
                return self._get_default_image()
            
            # Ensure data type is uint8
            if rgb_data.dtype != np.uint8:
                rgb_data = (rgb_data * 255).astype(np.uint8)
            
            return rgb_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get image from '{camera_name}' camera: {e}")
            return self._get_default_image()
    
    def _get_default_image(self):
        """Gets default image data."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _get_default_observation(self, task_description: str):
        """Gets default observation data."""
        return {
            "front": self._get_default_image(),
            "wrist": self._get_default_image(),
            "joint_pos": np.zeros(6, dtype=np.float32),
            "task_description": task_description
        }
    
    def execute_action(self, action: torch.Tensor):
        """
        Executes the action output by the VLA model.
        
        Args:
            action: The action tensor, shape (action_horizon, 6) or (action_horizon, 1, 6).
        """
        try:
            # Handle different action tensor shapes
            if len(action.shape) == 3:
                # Shape is (action_horizon, 1, 6)
                current_action = action[0, 0, :].cpu().numpy()  # (6,)
            elif len(action.shape) == 2:
                # Shape is (action_horizon, 6)
                current_action = action[0, :].cpu().numpy()  # (6,)
            else:
                logger.error(f"❌ Unsupported action tensor shape: {action.shape}")
                return
            
            # Directly set joint positions
            self.robot.set_joint_positions(current_action)
            
            logger.debug(f"✅ Executing action: {current_action}")
            
        except Exception as e:
            logger.error(f"❌ Failed to execute action: {e}")
    
    def reset_environment(self):
        """Resets the environment."""
        try:
            self.world.reset()
            logger.info("✅ Environment has been reset.")
        except Exception as e:
            logger.error(f"❌ Failed to reset environment: {e}")
    
    def step_environment(self, render: bool = True):
        """Steps the environment forward."""
        try:
            self.world.step(render=render)
        except Exception as e:
            logger.error(f"❌ Failed to step environment: {e}")
    
    def is_environment_ready(self) -> bool:
        """Checks if the environment is ready."""
        try:
            return self.world.is_playing()
        except Exception as e:
            logger.error(f"❌ Failed to check environment status: {e}")
            return False
