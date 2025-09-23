# -*- coding: utf-8 -*-
"""
IK Controller
Manages the robot arm's inverse kinematics (IK) calculations, posture correction,
and control logic.
"""

import numpy as np
import logging
from typing import Tuple, Optional

# Isaac Sim imports
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

logger = logging.getLogger(__name__)

class IKController:
    """
    Robot Arm IK Controller.
    Responsible for computing inverse kinematics, correcting posture, and sending
    joint commands to the robot.
    """
    
    def __init__(self, robot, config: dict, project_root: str):
        """
        Initializes the IKController.
        
        Args:
            robot: The robot articulation object.
            config: The configuration dictionary.
            project_root: The root path of the project.
        """
        self.robot = robot
        self.config = config
        self.project_root = project_root
        
        # Initial IK target position
        self.ik_target_position = np.array([0.25, 0.0, 0.25])
        
        # Default initial joint positions
        self.initial_joint_positions = np.array([0.0, 0.0, 0.0, np.radians(90), np.radians(-90), 0.0])
        
        # Initialize the IK solver
        self._setup_ik_solver()
        
        logger.info("✅ IKController initialized.")
        logger.info(f"   Initial IK target position: {self.ik_target_position.tolist()}")
        logger.info(f"   Initial joint positions (deg): {np.degrees(self.initial_joint_positions).tolist()}")
        
        # Cache for posture deviation, used by the state machine
        self._last_posture_deviation_deg = 0.0
        
        # Control flag for posture correction
        self.enable_posture_correction = False
    
    def _setup_ik_solver(self):
        """Sets up the LulaKinematicsSolver for IK calculations."""
        import os
        
        # Read paths from the configuration
        robot_config = self.config.get('robot', {})
        descriptor_path_rel = robot_config.get('descriptor_path')
        urdf_path_rel = robot_config.get('urdf_path')

        if not descriptor_path_rel or not urdf_path_rel:
            raise ValueError("'descriptor_path' or 'urdf_path' not found in the configuration file.")

        descriptor_path = os.path.join(self.project_root, descriptor_path_rel)
        urdf_path = os.path.join(self.project_root, urdf_path_rel)
        
        if not os.path.exists(descriptor_path):
            raise FileNotFoundError(f"IK descriptor file not found: {descriptor_path}")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        self.ik_solver = LulaKinematicsSolver(
            robot_description_path=descriptor_path,
            urdf_path=urdf_path
        )
        
        logger.info("✅ Lula IK solver initialized.")
        logger.info(f"   Descriptor file: {descriptor_path}")
        logger.info(f"   URDF file: {urdf_path}")
    
    def set_target_position(self, target_position: np.ndarray):
        """
        Sets the IK target position.
        
        Args:
            target_position: The target position [x, y, z].
        """
        old_position = getattr(self, 'ik_target_position', None)
        self.ik_target_position = np.array(target_position)
        
        # Log only when the position changes significantly to avoid spamming (threshold > 2cm)
        if old_position is None or np.linalg.norm(self.ik_target_position - old_position) > 0.02:
            logger.debug(f"Target position updated: {self.ik_target_position}")
    
    def get_target_position(self) -> np.ndarray:
        """Gets the current IK target position."""
        return np.copy(self.ik_target_position)
    
    def compute_ik(self, target_position: Optional[np.ndarray] = None, 
                   current_joint_positions: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Computes the inverse kinematics.
        
        Args:
            target_position: Target position. If None, uses the current IK target.
            current_joint_positions: Current joint positions for warm start.
            
        Returns:
            A tuple containing (joint_angles, success_flag).
        """
        if target_position is None:
            target_position = self.ik_target_position
        
        # Set robot base pose
        base_pos_w = np.array([0.0, 0.0, 0.0])
        base_quat_w = np.array([1.0, 0.0, 0.0, 0.0])
        self.ik_solver.set_robot_base_pose(base_pos_w, base_quat_w)
        
        # Prepare warm start
        if current_joint_positions is not None:
            warm_start = np.copy(current_joint_positions[:5])
            warm_start[3] = np.radians(90)
            warm_start[4] = np.radians(-90)
        else:
            warm_start = self.initial_joint_positions[:5]
        
        # Compute IK
        ik_sol_np, success = self.ik_solver.compute_inverse_kinematics(
            frame_name="wrist_link",
            target_position=target_position,
            warm_start=warm_start
        )
        
        if not success:
            # Throttle IK failure logs to once per second (approx. 60 frames) to avoid spam.
            if not hasattr(self, '_last_ik_failure_frame'):
                self._last_ik_failure_frame = 0
            if not hasattr(self, '_frame_counter'):
                self._frame_counter = 0
            
            self._frame_counter += 1
            if self._frame_counter - self._last_ik_failure_frame >= 60:
                logger.error(f"❌ IK solver failed for target position: {target_position.tolist()}")
                self._last_ik_failure_frame = self._frame_counter
            
            return np.zeros(5), False
        
        # Apply posture correction only when enabled (e.g., avoid during descent)
        if self.enable_posture_correction:
            ik_sol_np = self.apply_posture_correction(ik_sol_np)
        
        return ik_sol_np, True
    
    def apply_posture_correction(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Applies posture correction to keep the gripper pointing downwards.
        
        Args:
            joint_positions: The original joint positions from the IK solver.
            
        Returns:
            The corrected joint positions.
        """
        # Get joint limits
        lim_low, lim_high = self.ik_solver.get_cspace_position_limits()
        low_flex, high_flex = lim_low[3], lim_high[3]
        
        # Compute forward kinematics to get the end-effector pose
        ee_pos_uncorrected, ee_rot_uncorrected = self.ik_solver.compute_forward_kinematics(
            frame_name="wrist_link", 
            joint_positions=joint_positions
        )
        
        # --- Posture Correction Algorithm ---
        # This algorithm ensures the gripper's local Y-axis (forward direction)
        # aligns with the world's Z-axis (downward direction).
        
        # Current forward direction of the gripper (local -Y axis in world frame)
        gripper_forward_vec = ee_rot_uncorrected @ np.array([0, -1, 0])
        
        # Desired direction: vertically downwards
        desired_vec = np.array([0, 0, -1])
        
        # Calculate the angle error
        cos_angle = np.clip(np.dot(gripper_forward_vec, desired_vec), -1.0, 1.0)
        angle_error_rad = np.arccos(cos_angle)
        
        # Determine the rotation axis and correction direction
        rotation_axis = np.cross(gripper_forward_vec, desired_vec)
        wrist_flex_axis = ee_rot_uncorrected @ np.array([1, 0, 0])
        correction_sign = -np.sign(np.dot(rotation_axis, wrist_flex_axis))
        
        # Apply the posture correction
        corrected_positions = np.copy(joint_positions)
        base_angle_rad = np.radians(90.0)  # Use a fixed 90-degree base value
        correction_rad = correction_sign * angle_error_rad * 1.0
        final_flex_angle = np.clip(base_angle_rad + correction_rad, low_flex, high_flex)
        corrected_positions[3] = final_flex_angle
        
        # Cache posture deviation for the state machine
        self._last_posture_deviation_deg = np.degrees(angle_error_rad)
        
        return corrected_positions
    
    def set_posture_correction_enabled(self, enabled: bool):
        """Enables or disables the posture correction logic."""
        self.enable_posture_correction = enabled
        if enabled:
            logger.info("✅ Posture correction has been enabled.")
        else:
            logger.info("❌ Posture correction has been disabled.")
    
    def compute_forward_kinematics(self, joint_positions: np.ndarray, 
                                 frame_name: str = "wrist_link") -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes forward kinematics.
        
        Args:
            joint_positions: The joint positions.
            frame_name: The name of the target frame.
            
        Returns:
            A tuple containing (position, rotation_matrix).
        """
        return self.ik_solver.compute_forward_kinematics(
            frame_name=frame_name,
            joint_positions=joint_positions
        )
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the joint limits."""
        return self.ik_solver.get_cspace_position_limits()
    
    def move_to_position(self, target_position: np.ndarray, 
                        gripper_position: float = 0.0) -> bool:
        """
        Moves the robot to a specified target position.
        
        Args:
            target_position: The target end-effector position.
            gripper_position: The target gripper position.
            
        Returns:
            True if the command was sent successfully, False otherwise.
        """
        # Get current robot joint positions
        try:
            current_joint_pos = self.robot.get_joint_positions()
        except AttributeError:
            # Fallback to default if the method doesn't exist
            current_joint_pos = self.initial_joint_positions
            logger.warning("Robot object lacks get_joint_positions method; using default joint positions.")
        
        # Compute IK
        ik_sol, success = self.compute_ik(target_position, current_joint_pos)
        if not success:
            return False
        
        # Apply posture correction
        ik_sol_corrected = self.apply_posture_correction(ik_sol)
        
        # Prepare the final joint targets
        final_joint_targets = np.copy(current_joint_pos)
        final_joint_targets[:5] = ik_sol_corrected
        final_joint_targets[5] = gripper_position
        
        # Send the command
        articulation_controller = self.robot.get_articulation_controller()
        action = ArticulationAction()
        action.joint_positions = final_joint_targets
        articulation_controller.apply_action(action)
        
        logger.debug(f"✅ Robot command sent for target position: {target_position.tolist()}")
        return True
    
    def move_to_initial_position(self) -> bool:
        """Moves the robot to its initial position."""
        initial_target = np.array([0.25, 0.0, 0.25])
        return self.move_to_position(initial_target)
    
    def is_at_target(self, tolerance: float = 0.01) -> bool:
        """
        Checks if the end-effector has reached the target position.
        
        Args:
            tolerance: The tolerance in meters.
            
        Returns:
            True if the end-effector is within tolerance of the target.
        """
        try:
            # Get current joint positions from observations
            obs = self.robot.get_observations()
            robot_state = obs[self.robot.name] if hasattr(obs, "__getitem__") and self.robot.name in obs else {}
            current_joint_pos = robot_state.get("joint_positions", self.initial_joint_positions)
            
            # Compute the current end-effector position
            current_ee_pos, _ = self.compute_forward_kinematics(current_joint_pos[:5])
            
            # Calculate the distance to the target
            distance = np.linalg.norm(current_ee_pos - self.ik_target_position)
            return distance < tolerance
            
        except Exception as e:
            logger.error(f"❌ Error while checking target position: {e}")
            return False
    
    def get_posture_deviation_deg(self) -> float:
        """
        Calculates the angular deviation (in degrees) of the gripper from the vertical direction.
        
        Returns:
            The angular deviation in degrees (0 means perfectly vertical).
        """
        try:
            # Get current joint positions from observations
            obs = self.robot.get_observations()
            robot_state = obs[self.robot.name] if hasattr(obs, "__getitem__") and self.robot.name in obs else {}
            current_joint_pos = robot_state.get("joint_positions", self.initial_joint_positions)
            
            # Compute end-effector rotation matrix (for the wrist_link frame)
            _, ee_rot = self.compute_forward_kinematics(current_joint_pos[:5], frame_name="wrist_link")
            
            # Calculate the ray direction
            direction_local_red = np.array([0, -1, 0])  # Direction in the gripper's local coordinate system
            ray_direction_red = ee_rot @ direction_local_red  # Transform to world coordinates
            
            # Calculate the angular deviation from the vertical direction
            desired_vec_vertical = np.array([0, 0, -1])  # Desired vertically downward direction
            cos_angle_dev = np.clip(np.dot(ray_direction_red, desired_vec_vertical), -1.0, 1.0)
            angle_dev_rad = np.arccos(cos_angle_dev)
            angle_deviation_deg = np.degrees(angle_dev_rad)
            
            return angle_deviation_deg
            
        except Exception as e:
            logger.error(f"❌ Error calculating posture deviation: {e}")
            return 180.0  # Return max deviation as an error indicator
    
    def get_current_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the current end-effector pose.
        
        Returns:
            A tuple containing (position, rotation_matrix).
        """
        try:
            obs = self.robot.get_observations()
            robot_state = obs[self.robot.name] if hasattr(obs, "__getitem__") and self.robot.name in obs else {}
            current_joint_pos = robot_state.get("joint_positions", self.initial_joint_positions)
            
            return self.compute_forward_kinematics(current_joint_pos[:5])
        except Exception as e:
            logger.error(f"❌ Error getting end-effector pose: {e}")
            return np.zeros(3), np.eye(3)
    
    def set_target_position(self, target_position):
        """Sets the target position (for state machine use)."""
        old_position = getattr(self, 'ik_target_position', None)
        self.ik_target_position = np.array(target_position)
        
        # Log only when the position changes significantly to avoid spamming (threshold > 2cm)
        if old_position is None or np.linalg.norm(self.ik_target_position - old_position) > 0.02:
            logger.debug(f"Target position updated: {self.ik_target_position}")
    
    @property
    def current_target_position(self):
        """Gets the current target position (for state machine use)."""
        return self.ik_target_position.copy()
    
    def get_posture_deviation_deg(self):
        """Gets the cached posture deviation angle (for state machine use)."""
        return self._last_posture_deviation_deg
    
    def get_posture_deviation_deg_from_joints(self, joint_positions):
        """
        Calculates the posture deviation angle (in degrees) from a given set of joint positions.
        
        Args:
            joint_positions: An array of joint positions.
            
        Returns:
            The posture deviation angle in degrees.
        """
        # Compute forward kinematics
        ee_pos, ee_rot = self.compute_forward_kinematics(
            frame_name="wrist_link", joint_positions=joint_positions
        )
        
        # Calculate gripper forward vector
        gripper_forward_vec = ee_rot @ np.array([0, -1, 0])
        
        # Desired vertically downward vector
        desired_vec = np.array([0, 0, -1])
        
        # Calculate angle deviation
        cos_angle = np.clip(np.dot(gripper_forward_vec, desired_vec), -1.0, 1.0)
        angle_error_rad = np.arccos(cos_angle)
        
        return np.degrees(angle_error_rad)
    
    def set_initial_joint_positions(self, robot):
        """Directly sets the initial joint positions, bypassing IK - used for IK failure recovery."""
        # Use the standard initial joint positions
        initial_joint_positions = np.array([0.0, 0.0, 0.0, np.radians(90), np.radians(-90), 0.0])
        
        try:
            # Directly set joint positions, bypassing the IK solver
            articulation_controller = robot.get_articulation_controller()
            
            # Create an articulation action
            from isaacsim.core.utils.types import ArticulationAction
            action = ArticulationAction()
            
            # Get current joint positions and set the new target
            current_joint_positions = robot.get_joint_positions()
            target_joint_positions = current_joint_positions.copy()
            target_joint_positions[:6] = initial_joint_positions  # First 6 joints
            
            action.joint_positions = target_joint_positions
            articulation_controller.apply_action(action)
            
            logger.info("✅ Successfully set initial joint positions, bypassing IK.")
            logger.info(f"   Joint positions: {initial_joint_positions}")
            
            # Also update the IK target to a reachable position
            self.ik_target_position = np.array([0.25, 0.0, 0.25])
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set initial joint positions: {e}")
            return False
    
    def execute_control(self, robot, state_machine=None) -> bool:
        """
        Executes the full robot control loop: IK calculation, posture correction, and command dispatch.
        Handles IK failures gracefully.
        
        Args:
            robot: The robot articulation object.
            state_machine: The state machine object, used for IK failure recovery.
            
        Returns:
            True if the control was successful, False otherwise.
        """
        try:
            # Get current joint positions
            current_joint_positions = robot.get_joint_positions()
            
            # Compute IK
            ik_solution, success = self.compute_ik(current_joint_positions=current_joint_positions)
            if not success:
                # Throttle error logging to avoid spam
                if not hasattr(self, '_last_ik_failure_frame'):
                    self._last_ik_failure_frame = 0
                if not hasattr(self, '_frame_counter'):
                    self._frame_counter = 0
                
                self._frame_counter += 1
                if self._frame_counter - self._last_ik_failure_frame >= 60:
                    logger.error(f"❌ IK solver failed for target: {self.ik_target_position.tolist()}")
                    logger.error("❌ IK solution not found. Halting current task and marking as failed.")
                    
                    # Print detailed failure information, including current arm position
                    try:
                        current_ee_pos, current_ee_rot = self.compute_forward_kinematics(current_joint_positions, "wrist_link")
                        print("❌ IK solution not found. Halting current task.")
                        print(f"🔍 IK Failure Details:")
                        print(f"   Target Position: [{self.ik_target_position[0]:.4f}, {self.ik_target_position[1]:.4f}, {self.ik_target_position[2]:.4f}]")
                        print(f"   Current EE Position: [{current_ee_pos[0]:.4f}, {current_ee_pos[1]:.4f}, {current_ee_pos[2]:.4f}]")
                        print(f"   Current Joint Positions (deg): {np.degrees(current_joint_positions[:5]).tolist()}")
                        print(f"   Position Error: {np.linalg.norm(self.ik_target_position - current_ee_pos):.4f}m")
                    except Exception as e:
                        print(f"❌ IK solution not found; could not retrieve detailed info: {e}")
                    
                    # Immediately trigger the state machine's IK failure recovery
                    if state_machine is not None:
                        state_machine.handle_ik_failure()
                    
                    self._last_ik_failure_frame = self._frame_counter
                
                return False
            
            # If IK succeeds, reset any failure counters
            self._ik_failure_count = 0
            
            # Apply posture correction
            corrected_solution = self.apply_posture_correction(ik_solution)
            
            # Update the gripper controller state
            if state_machine is not None:
                state_machine.gripper_controller.update_gripper_position()
            
            # Prepare the complete joint target array (including gripper)
            target_joint_positions = np.copy(current_joint_positions)
            target_joint_positions[:5] = corrected_solution  # First 5 arm joints
            
            # Set gripper position
            if state_machine is not None:
                target_joint_positions[5] = state_machine.gripper_controller.get_target_position()
            
            # Send the command to the robot
            from isaacsim.core.utils.types import ArticulationAction
            action = ArticulationAction()
            action.joint_positions = target_joint_positions
            
            articulation_controller = robot.get_articulation_controller()
            articulation_controller.apply_action(action)
            
            return True
            
        except Exception as e:
            logger.warning(f"Control execution failed: {e}")
            return False




