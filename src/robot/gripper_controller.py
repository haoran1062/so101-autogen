# -*- coding: utf-8 -*-
"""
Gripper Controller
Manages the state and movement of the robot's gripper.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class GripperController:
    """
    A progressive controller for the gripper.
    It manages the target position and applies incremental changes.
    """
    
    def __init__(self, open_pos: float, closed_pos: float):
        """
        Initializes the GripperController.
        
        Args:
            open_pos: The position when the gripper is fully open.
            closed_pos: The position when the gripper is fully closed.
        """
        self.open_pos = open_pos
        self.closed_pos = closed_pos
        
        # Control parameters
        self.gripper_step = 0.01
        self.current_gripper_position = open_pos
        self.target_gripper_position = open_pos
        self.gripper_delta = 0.0
        
        logger.info(f"✅ Progressive GripperController initialized.")
        logger.info(f"   Open/Close Range: {open_pos:.3f} to {closed_pos:.3f}")
        logger.info(f"   Step size: {self.gripper_step:.3f}")
    
    def get_target_position(self) -> float:
        """Gets the target gripper position."""
        return self.target_gripper_position
    
    def get_current_position(self) -> float:
        """Gets the current gripper position."""
        return self.current_gripper_position
    
    def set_target_position(self, position: float):
        """
        Sets the target position.
        
        Args:
            position: The target position, clamped within the open/closed range.
        """
        self.target_gripper_position = max(min(position, self.open_pos), self.closed_pos)
        logger.debug(f"🎯 Gripper target position set to: {self.target_gripper_position:.3f}")
    
    def update_gripper_position(self):
        """
        Updates the gripper position by applying the accumulated delta command.
        """
        self.target_gripper_position += self.gripper_delta
        
        # Clamp within the open/closed range
        if self.target_gripper_position < self.closed_pos:
            self.target_gripper_position = self.closed_pos
        elif self.target_gripper_position > self.open_pos:
            self.target_gripper_position = self.open_pos
        
        self.current_gripper_position = self.target_gripper_position
        
        # Reset delta
        self.gripper_delta = 0.0
    
    def start_opening(self):
        """
        Initiates the opening action (e.g., on key press).
        """
        self.gripper_delta = self.gripper_step
        logger.debug("🔓 Starting to open gripper.")
    
    def stop_opening(self):
        """
        Stops the opening action (e.g., on key release).
        """
        self.gripper_delta = 0.0
        logger.debug("⏹️ Stopped opening gripper.")
    
    def start_closing(self):
        """
        Initiates the closing action (e.g., on key press).
        """
        self.gripper_delta = -self.gripper_step
        logger.debug("🔒 Starting to close gripper.")
    
    def stop_closing(self):
        """
        Stops the closing action (e.g., on key release).
        """
        self.gripper_delta = 0.0
        logger.debug("⏹️ Stopped closing gripper.")
    
    def open_gripper(self):
        """Fully opens the gripper."""
        self.target_gripper_position = self.open_pos
        self.current_gripper_position = self.open_pos
        logger.info("🔓 Gripper fully opened.")
    
    def close_gripper(self, percentage: float = 1.0):
        """
        Closes the gripper to a specified percentage.
        
        Args:
            percentage: The closing percentage (0.0 = fully open, 1.0 = fully closed).
        """
        percentage = max(0.0, min(1.0, percentage))
        self.target_gripper_position = self.open_pos + (self.closed_pos - self.open_pos) * percentage
        self.current_gripper_position = self.target_gripper_position
        logger.info(f"🔒 Gripper closed to {percentage*100:.1f}%.")
    
    def close_to_grasp_position(self):
        """
        Closes the gripper to a predefined grasp position (e.g., 26.5% open).
        """
        grasp_percentage = 0.265  # 26.5% open
        self.target_gripper_position = self.closed_pos + (self.open_pos - self.closed_pos) * grasp_percentage
        self.current_gripper_position = self.target_gripper_position
        logger.info(f"🤏 Gripper set to grasp position (26.5% open): {self.target_gripper_position:.3f}")
    
    def is_fully_open(self, tolerance: float = 0.001) -> bool:
        """
        Checks if the gripper is fully open.
        
        Args:
            tolerance: The tolerance for the check.
            
        Returns:
            True if the gripper is fully open, False otherwise.
        """
        return abs(self.current_gripper_position - self.open_pos) < tolerance
    
    def is_fully_closed(self, tolerance: float = 0.001) -> bool:
        """
        Checks if the gripper is fully closed.
        
        Args:
            tolerance: The tolerance for the check.
            
        Returns:
            True if the gripper is fully closed, False otherwise.
        """
        return abs(self.current_gripper_position - self.closed_pos) < tolerance
    
    def get_openness_percentage(self) -> float:
        """
        Gets the gripper's openness percentage.
        
        Returns:
            The openness percentage (0.0 = fully closed, 1.0 = fully open).
        """
        range_total = self.open_pos - self.closed_pos
        if range_total == 0:
            return 1.0
        
        current_from_closed = self.current_gripper_position - self.closed_pos
        return current_from_closed / range_total
    
    def __repr__(self) -> str:
        """String representation of the GripperController."""
        return (f"GripperController(current={self.current_gripper_position:.3f}, "
                f"target={self.target_gripper_position:.3f}, "
                f"openness={self.get_openness_percentage()*100:.1f}%)")
