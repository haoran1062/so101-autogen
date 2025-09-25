# -*- coding: utf-8 -*-
"""
Simplified Grasping State Machine.
Derived from a reference script's state machine, this version removes complex
logic such as searching, confirmation, and recovery for a more streamlined process.
"""

import numpy as np
import logging
import random
from datetime import datetime
from .grasp_states import SimpleGraspingState

logger = logging.getLogger(__name__)

class SimpleGraspingStateMachine:
    """
    A simplified state machine for grasping tasks.
    
    Features:
    - Linear state flow without branching or searching.
    - Fully automated without user confirmation steps.
    - Proceeds directly to descent if posture adjustment fails.
    - Does not retry on grasp failure; marks the task as failed immediately.
    - Automatically returns to the initial position if an IK solution is not found.
    """
    
    def __init__(self, world, robot, ik_controller, gripper_controller, 
                 pickup_assessor, scene_manager, target_configs, draw_interface=None,
                 data_collection_manager=None, camera_controller=None):
        """
        Initializes the simplified state machine.
        
        Args:
            world: The Isaac Sim world object.
            robot: The robot arm object.
            ik_controller: The IK controller.
            gripper_controller: The gripper controller.
            pickup_assessor: The grasp assessor.
            scene_manager: The scene manager.
            target_configs: A dictionary of target configurations.
            draw_interface: The drawing interface (optional).
            data_collection_manager: The data collection manager (optional).
            camera_controller: The camera controller (optional).
        """
        # Core components
        self.world = world
        self.robot = robot
        self.ik_controller = ik_controller
        self.gripper_controller = gripper_controller
        self.pickup_assessor = pickup_assessor
        self.scene_manager = scene_manager
        self.target_configs = target_configs
        self.draw = draw_interface
        
        # Data collection components
        self.data_collection_manager = data_collection_manager
        self.camera_controller = camera_controller
        self.episode_id_counter = 0
        
        # Grasp detector (parameters read from config)
        from src.robot.grasp_detector import GraspDetector
        grasp_config = self.scene_manager.config.get('grasp_detection', {})
        self.grasp_detector = GraspDetector(grasp_config)
        
        # Smart placement manager
        from src.robot.smart_placement_manager import SmartPlacementManager
        placement_config = self.scene_manager.config.get('placement', {})
        # Get plate info from the plate config section
        plate_config = self.scene_manager.config.get('scene', {}).get('plate', {})
        placement_config.update({
            'plate_position': plate_config.get('position', [0.28, -0.05, 0.02]),
            'plate_radius': plate_config.get('virtual_config', {}).get('radius', 0.10),
            'plate_height': plate_config.get('virtual_config', {}).get('height', 0.02),
        })
        # Read placement limits from the config
        placement_limits = self.scene_manager.config.get('placement', {}).get('placement_limits', {})
        placement_config.update({
            'max_x_distance': placement_limits.get('max_x_distance', 0.30),
            'max_y_distance': placement_limits.get('max_y_distance', 0.20),
        })
        self.placement_manager = SmartPlacementManager(placement_config)
        
        # State machine status
        self.current_state = SimpleGraspingState.IDLE
        self.target_prim_path = None
        self.target_prim_name = None
        self.target_object = None
        self.last_attempt_successful = False # For automation scripts to query
        self._monitoring_plate_initial_pos = None # To monitor plate position
        self.plate_name_for_monitoring = None # To get the plate object in real-time
        self.hard_reset_required = False # Flag to request a hard scene reset if the plate moves
        
        # Timers and counters
        self.state_timer = 0
        self.retry_count = 0
        self.frame_count = 0 # Frame counter
        
        # Read state machine control parameters from config
        sm_config = self.scene_manager.config.get('state_machine_control', {})
        grasp_config = sm_config.get('grasping', {})
        move_config = sm_config.get('movement_speeds', {})

        # Movement control
        self.is_moving = False
        self.move_start_pos = None
        self.move_end_pos = None
        self.move_progress = 0.0
        self.move_duration_steps = 90
        
        # State configuration
        self.max_posture_adjust_steps = 180   # Max steps for posture adjustment (3s @ 60FPS)
        self.max_descend_steps = 600          # Max steps for descent (10s) - increased for slower descent
        self.grasp_duration_steps = int(grasp_config.get('close_duration_s', 1.5) * 60) # Grasp duration
        self.grasp_settle_duration_steps = int(grasp_config.get('settle_duration_s', 0.5) * 60) # Grasp settle time
        self.release_duration_steps = 180     # Release duration (6s) - more time for the object to stabilize
        
        # Load speed parameters from config
        self.travel_horizontal_speed = move_config.get('travel_horizontal_step_m', 0.0025)
        self.descend_step_size = move_config.get('descend_step_m', 0.0015)
        self.lift_step_size = move_config.get('lift_step_m', 0.002)
        
        # Gripper close angle random range
        self.close_angle_min = grasp_config.get('close_angle_percent_min', 0.225)
        self.close_angle_max = grasp_config.get('close_angle_percent_max', 0.225)

        # Thresholds
        self.posture_threshold_deg = 5.0      # Posture adjustment angle threshold (degrees)
        self.approach_height = 0.25           # Approach height
        self.grasp_height_offset = 0.015      # Grasp height offset
        self.lift_height = 0.20              # Lift height - lift to 20cm first to avoid unsolvable IK
        self.safe_height = 0.30              # Safe position height
        self.transport_height = 0.25         # Transport height - 25cm to avoid collisions
        self.release_height = 0.21           # Release height
        self.initial_position = np.array([0.25, 0.0, 0.25])  # Initial position
        
        # Gripper control
        self.grasp_start_pos = 0.0
        self.grasp_end_pos = 0.0
        
        # Initial joint positions (degrees)
        self.initial_joint_positions = np.array([0.0, 0.0, 0.0, 90.0, -90.0, 0.0])
        
        logger.info("✅ Simplified grasping state machine initialized.")
        logger.info(f"    Initial State: {self.current_state.get_display_name()}")
        logger.info(f"    Target Configurations: {len(self.target_configs)}")
        
        # Ensure posture correction is disabled initially
        self.ik_controller.set_posture_correction_enabled(False)
        
        # Ensure gripper is open on initialization
        self.gripper_controller.set_target_position(self.gripper_controller.open_pos)
        print("🤏 State machine initialized: Gripper set to open.")
        logger.info("State machine initialized: Gripper set to open.")
        
        # Read plate movement monitoring config
        self.plate_monitoring_config = self.scene_manager.config.get('plate_monitoring', {})
        self.is_plate_monitoring_enabled = self.plate_monitoring_config.get('enabled', False)
        if self.is_plate_monitoring_enabled:
            logger.info("🛡️ Plate movement monitoring enabled.")
            logger.info(f"   Position Threshold: {self.plate_monitoring_config.get('position_threshold', 0.03)}m")
            logger.info(f"   Velocity Threshold: {self.plate_monitoring_config.get('velocity_threshold', 0.1)}m/s")

    def is_busy(self):
        """Checks if the state machine is currently executing a task."""
        return self.current_state not in [
            SimpleGraspingState.IDLE, 
            SimpleGraspingState.SUCCESS, 
            SimpleGraspingState.FAILED
        ]
        
    def get_last_attempt_status(self):
        """Gets the result of the last grasp attempt."""
        return self.last_attempt_successful

    def get_current_state(self):
        """Gets the current state as an enum name."""
        return self.current_state.name
        
    def calculate_safe_position(self, object_pos):
        """Calculates a safe position by moving 2/7 of the way from the object towards the origin."""
        # Check if the object is too close to the origin
        distance_from_origin = np.linalg.norm(object_pos[:2])
        if distance_from_origin < 0.05:  # If distance is less than 5cm
            print(f"⚠️ Object is too close to origin ({distance_from_origin:.4f}m), using default safe position.")
            # Use a default safe position to avoid unsolvable IK
            safe_pos = np.array([0.15, 0.0, self.safe_height])
            print(f"📍 Using default safe position: {safe_pos}")
            return safe_pos
        
        # Move 2/7 of the way towards the origin in the XY plane (i.e., multiply by 5/7)
        safe_xy = object_pos[:2] * (5.0 / 7.0)
        
        # Use the safe height for the Z coordinate
        safe_z = self.safe_height
        
        safe_pos = np.array([safe_xy[0], safe_xy[1], safe_z])
        print(f"📍 Safe position calculated: Object at {object_pos[:2]} -> moved 2/7 towards origin -> {safe_pos}")
        return safe_pos
        
    def start_grasp_sequence(self, target_key):
        """
        Starts the grasp sequence.
        
        Args:
            target_key (str): The number key pressed ("1", "2", "3").
        """
        # Validate input
        if not target_key.isdigit():
            print(f"❌ Invalid input: {target_key}. Please press a number key to select a target.")
            return False
            
        # Reset the result of the previous grasp attempt
        self.last_attempt_successful = False
        
        target_index = int(target_key) - 1
        target_paths = list(self.target_configs.keys())
        
        if target_index < 0 or target_index >= len(target_paths):
            print(f"❌ Target index out of range: {target_index+1}/{len(target_paths)}")
            return False
            
        # Set the target
        self.target_prim_path = target_paths[target_index]
        self.target_prim_name = self.target_configs[self.target_prim_path]["name"]
        
        # Reset posture adjustment counters
        self._posture_adjust_attempts = 0
        self._posture_adjust_wait_frames = 0
        print(f"🔄 Resetting posture adjustment counters for new grasp task.")
        
        # Get the target object
        try:
            self.target_object = self.world.scene.get_object(self.target_prim_name)
            if self.target_object is None:
                print(f"❌ Could not find target object: {self.target_prim_name}")
                return False
        except Exception as e:
            print(f"❌ Failed to get target object: {e}")
            return False
        
        # Scan for existing oranges in the scene to update the placement manager
        self._scan_existing_oranges()
        
        # Set the target object for the grasp detector
        self.grasp_detector.set_target_object(self.target_object)
        self.grasp_detector.reset_detection()  # Reset detection state
        print(f"🔍 Grasp detector target set to: {self.target_prim_name}")
        
        # Set up the plate object (from config)
        self._setup_plate_object()
            
        # Start data collection (if enabled)
        if self.data_collection_manager:
            self.episode_id_counter += 1
            episode_id = f"episode_{self.episode_id_counter:04d}"
            target_info = {
                'name': self.target_prim_name,
                'position': self.target_object.get_world_pose()[0].tolist() if self.target_object else None
            }
            self.data_collection_manager.start_episode(episode_id, target_info)
            print(f"📊 Data collection started for episode: {episode_id}")
        
        # Start the state machine
        print(f"\n🎯 Starting grasp sequence for: {self.target_prim_name}")
        print(f"📋 State flow: {SimpleGraspingState.get_state_flow_description()}")
        
        self._transition_to_state(SimpleGraspingState.APPROACH)
        return True
    
    def fail_current_task(self):
        """Externally called method to force the current task to fail (e.g., on timeout)."""
        if self.is_busy():
            print("⏰ Task was forced to fail externally (e.g., timeout).")
            self._return_to_initial_position()
            self._transition_to_state(SimpleGraspingState.FAILED)

    def _setup_plate_object(self):
        """Sets up the plate object from the configuration."""
        try:
            # Read plate info from config
            plate_config = self.scene_manager.config.get('scene', {}).get('plate', {})
            plate_name = plate_config.get('model', 'plate_object')
            plate_position = plate_config.get('position', [0.28, -0.05, 0.02])
            use_virtual = plate_config.get('use_virtual_plate', True)

            print("🍽️ Reading plate information from config:")
            print(f"    Name: {plate_name}")
            print(f"    Position: {plate_position}")
            print(f"    Use Virtual Object: {use_virtual}")

            # If plate movement monitoring is enabled, we must use the real physical plate
            if self.is_plate_monitoring_enabled:
                if use_virtual:
                    print("🛡️ Plate movement monitoring is enabled, forcing use of the real physical plate object.")
                    logger.info("Plate movement monitoring is enabled, forcing use of the real plate object.")
                use_virtual = False # Force virtual mode off

            if not use_virtual:
                # Find the actual loaded plate object from the scene manager
                found_plate_prim_name = None
                found_plate_object = None
                for prim_name, scene_obj in self.scene_manager.scene_objects.items():
                    if "plate" in prim_name.lower():
                        found_plate_prim_name = prim_name
                        found_plate_object = scene_obj
                        break # Stop after finding the first one

                if found_plate_object:
                    self.grasp_detector.set_plate_object(found_plate_object)
                    self.plate_name_for_monitoring = found_plate_prim_name # Record the real plate's name
                    print(f"🍽️ Successfully set real plate object: {found_plate_prim_name}")
                else:
                    print(f"⚠️ Could not find a real plate object in the scene manager. Falling back to a virtual object.")
                    self.plate_name_for_monitoring = None # Clear the name
                    # If monitoring is enabled but no real plate is found, disable monitoring and warn the user
                    if self.is_plate_monitoring_enabled:
                        print("    ‼️ WARNING: Plate movement monitoring has been automatically disabled for this task.")
                        logger.warning("Could not find a real plate; plate movement monitoring is disabled.")
                        self.is_plate_monitoring_enabled = False # Temporarily disable
                    
                    # Fallback to virtual plate
                    self.grasp_detector.set_plate_object(VirtualPlate(plate_position))
                    print(f"🍽️ Created a virtual plate object at position: {np.array(plate_position)}")
            else:
                self.grasp_detector.set_plate_object(VirtualPlate(plate_position))
                print(f"🍽️ Created a virtual plate object at position: {np.array(plate_position)}")

            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"❌ Failed to set up plate object: {e}. Using default configuration.")
            self._create_virtual_plate_object([0.28, -0.05, 0.02])
    
    def _create_virtual_plate_object(self, plate_position=None):
        """Creates a virtual plate object for placement detection."""
        try:
            # Use the provided position parameter
            if plate_position is not None:
                plate_pos = np.array(plate_position)
            elif hasattr(self.scene_manager, 'plate_position') and self.scene_manager.plate_position is not None:
                plate_pos = np.array(self.scene_manager.plate_position)
            else:
                plate_pos = np.array([0.28, -0.05, 0.02])  # Default plate position
            
            # Create a virtual plate object class
            class VirtualPlateObject:
                def __init__(self, position):
                    self.position = np.array(position)
                
                def get_world_pose(self):
                    return self.position, np.array([0, 0, 0, 1])  # position and quaternion
                
                def get_name(self):
                    return "virtual_plate"
            
            virtual_plate = VirtualPlateObject(plate_pos)
            self.grasp_detector.set_plate_object(virtual_plate)
            print(f"🍽️ Created a virtual plate object at position: {plate_pos}")
            
        except Exception as e:
            print(f"❌ Failed to create virtual plate object: {e}")
        
    def update(self):
        """Updates the state machine (should be called every frame)."""
        self.frame_count += 1
        self.state_timer += 1
        
        # Only monitor plate stability during critical phases when the robot is near the workbench
        MONITORING_ACTIVE_STATES = [
            SimpleGraspingState.DESCEND,
            SimpleGraspingState.GRASP,
            SimpleGraspingState.LIFT,
            SimpleGraspingState.TRANSPORT,
            SimpleGraspingState.RELEASE,
        ]
        if self.current_state in MONITORING_ACTIVE_STATES and self.is_plate_monitoring_enabled:
            if not self._check_plate_stability():
                # _check_plate_stability will call fail_current_task internally if it fails
                return

        # Record a data collection frame (if enabled and a grasp task is active)
        if self.data_collection_manager and self.current_state != SimpleGraspingState.IDLE:
            self._record_data_collection_frame()
        
        # Update smooth movement
        self._update_smooth_move()
        
        # Execute logic based on the current state
        if self.current_state == SimpleGraspingState.IDLE:
            self._update_idle_state()
        elif self.current_state == SimpleGraspingState.APPROACH:
            self._update_approach_state()
        elif self.current_state == SimpleGraspingState.POSTURE_ADJUST:
            self._update_posture_adjust_state()
        elif self.current_state == SimpleGraspingState.DESCEND:
            self._update_descend_state()
        elif self.current_state == SimpleGraspingState.GRASP:
            self._update_grasp_state()
        elif self.current_state == SimpleGraspingState.GRASP_SETTLE:
            self._update_grasp_settle_state()
        elif self.current_state == SimpleGraspingState.LIFT:
            self._update_lift_state()
        elif self.current_state == SimpleGraspingState.RETREAT:
            self._update_retreat_state()
        elif self.current_state == SimpleGraspingState.TRANSPORT:
            self._update_transport_state()
        elif self.current_state == SimpleGraspingState.RELEASE:
            self._update_release_state()
        elif self.current_state == SimpleGraspingState.RETURN_HOME:
            self._update_return_home_state()
        elif self.current_state in [SimpleGraspingState.SUCCESS, SimpleGraspingState.FAILED]:
            self._update_terminal_state()
            
    def _transition_to_state(self, new_state):
        """Transitions to a new state."""
        old_state = self.current_state
        self.current_state = new_state
        self.state_timer = 0
        
        print(f"🔄 State transition: {old_state.get_display_name()} -> {new_state.get_display_name()}")
        logger.info(f"State transition: {old_state} -> {new_state}")
        
        # Clean up state-related cached variables
        if hasattr(self, '_retreat_safe_position'):
            delattr(self, '_retreat_safe_position')
        if hasattr(self, '_transport_final_position'):
            delattr(self, '_transport_final_position')
        if hasattr(self, '_return_home_started'):
            delattr(self, '_return_home_started')
        
        # Initialize the new state
        self._on_enter_state(new_state)
        
    def _on_enter_state(self, state):
        """Initialization logic for entering a new state."""
        if state == SimpleGraspingState.APPROACH:
            self._start_approach()
        elif state == SimpleGraspingState.POSTURE_ADJUST:
            print("📐 Starting posture adjustment...")
            # Stop movement and start posture adjustment
            self.is_moving = False
            # Initialize wait frames to 0 to ensure adjustment can run on the first frame
            self._posture_adjust_wait_frames = 0
        elif state == SimpleGraspingState.DESCEND:
            # Activate monitoring just before descent, when the scene should be stable
            if self.is_plate_monitoring_enabled:
                print("🛡️ Plate monitoring activated.")
                plate_object = self.grasp_detector.get_plate_object()
                if plate_object and hasattr(plate_object, 'get_world_pose'):
                    pos, _ = plate_object.get_world_pose()
                    self._monitoring_plate_initial_pos = np.array(pos)
                    logger.info(f"🛡️ Recorded stable plate position for monitoring: {self._monitoring_plate_initial_pos}")
                else:
                    logger.warning("⚠️ Could not record stable plate position; monitoring will be skipped for this run.")
                    self._monitoring_plate_initial_pos = None

            self._start_descend()
        elif state == SimpleGraspingState.GRASP:
            self._start_grasp()
        elif state == SimpleGraspingState.LIFT:
            self._start_lift()
        elif state == SimpleGraspingState.TRANSPORT:
            self._start_transport()
        elif state == SimpleGraspingState.RELEASE:
            self._start_release()
        elif state == SimpleGraspingState.RETREAT:
            pass # Logic is now handled entirely within _update_retreat_state()
        elif state in [SimpleGraspingState.SUCCESS, SimpleGraspingState.FAILED]:
            self._handle_terminal_state(state)
            
    def _start_approach(self):
        """Starts approaching the target."""
        if self.target_object is None:
            self._transition_to_state(SimpleGraspingState.FAILED)
            return
        
        # Ensure the gripper is open
        self.gripper_controller.set_target_position(self.gripper_controller.open_pos)
        print("🤏 Gripper opened, ready to grasp.")
        
        try:
            target_pos, _ = self.target_object.get_world_pose()
            approach_pos = np.array([target_pos[0], target_pos[1], self.approach_height])
            self._start_smooth_move(approach_pos, self.travel_horizontal_speed)
            print(f"🚀 Approaching target: {target_pos[:2]} -> Height {self.approach_height}m")
        except Exception as e:
            print(f"❌ Failed to approach target: {e}")
            self._transition_to_state(SimpleGraspingState.FAILED)
            
    def _start_descend(self):
        """Starts the slow descent."""
        print("⬇️ Starting slow descent, checking for green state frame by frame...")
        # Not using smooth move; will be handled frame by frame
            
    def _start_grasp(self):
        """Starts the grasp."""
        self.is_moving = False  # Stop any movement
        
        # Set gripper grasp parameters
        open_pos = self.gripper_controller.open_pos
        closed_pos = self.gripper_controller.closed_pos
        
        self.grasp_start_pos = self.gripper_controller.get_target_position()
        
        # Use a random close angle
        random_close_percent = random.uniform(self.close_angle_min, self.close_angle_max)
        self.grasp_end_pos = closed_pos + (open_pos - closed_pos) * random_close_percent
        
        print(f"🤏 Starting grasp, holding position at green state, closing gripper to {random_close_percent*100:.1f}% openness...")
        print(f"    Grasp Position: {self.ik_controller.current_target_position[:3]}")
        print(f"    Grasp Duration: {self.grasp_duration_steps / 60.0:.2f} seconds")
        
    def _start_lift(self):
        """Starts the slow lift."""
        print("⬆️ Starting slow lift, checking if object was successfully grasped...")
        # Not using smooth move; will be handled frame by frame
        
        # Set lift parameters
        self.lift_check_interval = 30  # Check grasp status every 30 frames
        
    def _start_transport(self):
        """Starts transport to the plate's position."""
        # Get the real plate position
        if hasattr(self.scene_manager, 'plate_position') and self.scene_manager.plate_position is not None:
            plate_pos = np.array(self.scene_manager.plate_position)
            plate_pos[2] = self.transport_height  # Use the transport height
            print(f"✅ Using real plate position: {self.scene_manager.plate_position}")
        else:
            # Fallback to the default position from config
            plate_pos = np.array([0.28, -0.05, self.transport_height])
            print(f"⚠️ Using default plate position: {plate_pos}")
            
        self._start_smooth_move(plate_pos, self.travel_horizontal_speed)
        print(f"🚚 Transporting to above plate: [{plate_pos[0]:.3f}, {plate_pos[1]:.3f}, {plate_pos[2]:.3f}]")
        print(f"    Transport Height: {self.transport_height}m (25cm to avoid collisions)")
        
    def _start_release(self):
        """Starts the release."""
        self.is_moving = False
        self.gripper_controller.target_gripper_position = self.gripper_controller.open_pos
        print("🖐️ Opening gripper to release object.")
        
    def _return_to_initial_position(self):
        """Returns to the initial position, bypassing IK by setting joint positions directly."""
        print("🔄 Returning robot arm to initial position...")
        logger.info("Returning robot arm to initial position.")
        
        # Directly set joint positions to bypass IK calculation (and avoid potential unsolvable IK loops)
        success = self.ik_controller.set_initial_joint_positions(self.robot)
        
        if success:
            print("✅ Robot arm has been returned to its initial joint positions.")
        else:
            print("⚠️ Failed to set joint positions directly. Attempting to move via IK.")
            # Fallback to IK if direct setting fails
            initial_ik_position = np.array([0.25, 0.0, 0.25])
            self.ik_controller.set_target_position(initial_ik_position)
        
        # Reset the gripper to the open state
        self.gripper_controller.set_target_position(self.gripper_controller.open_pos)
        
        print("🔄 Robot arm has returned to its initial position.")
        
    def _handle_terminal_state(self, state):
        """Handles terminal states."""
        if state == SimpleGraspingState.SUCCESS:
            print("\n🎉 Grasp task completed successfully!")
            self.last_attempt_successful = True
            # Handle successful episode completion
            self._handle_episode_end(success=True)
        else:
            print("\n❌ Grasp task failed.")
            self.last_attempt_successful = False
            # Handle failed episode completion
            self._handle_episode_end(success=False)
            # Immediately return to the initial position on failure
            self._return_to_initial_position()
            
        # Automatically return to the IDLE state after 2 seconds
        if self.state_timer > 120:
            self._reset_to_idle()
            
    def _reset_to_idle(self):
        """Resets the state machine to the IDLE state."""
        self.current_state = SimpleGraspingState.IDLE
        self.target_prim_path = None
        self.target_prim_name = None
        self.target_object = None
        self.state_timer = 0
        self.is_moving = False
        self._monitoring_plate_initial_pos = None # Clear snapshot
        self.plate_name_for_monitoring = None # Clear plate name
        self.hard_reset_required = False # Reset the hard reset request
        
        # Reset the gripper to the open state
        self.gripper_controller.target_gripper_position = self.gripper_controller.open_pos
        
        print("\n🔄 State machine has been reset. Waiting for new grasp command...")
        logger.info("State machine has been reset to the IDLE state.")
        
    def _start_smooth_move(self, end_pos, speed):
        """Starts a smooth movement, with duration calculated dynamically based on speed."""
        self.move_start_pos = np.copy(self.ik_controller.current_target_position)
        self.move_end_pos = end_pos
        self.move_progress = 0.0
        
        # Dynamically calculate movement duration
        distance = np.linalg.norm(self.move_end_pos - self.move_start_pos)
        if speed <= 1e-6: # Avoid division by zero
            self.move_duration_steps = 1
        else:
            self.move_duration_steps = max(1, int(distance / speed)) # Must be at least 1 step
            
        self.is_moving = True
        
        logger.debug(f"Starting smooth move: {self.move_start_pos} -> {self.move_end_pos} ({self.move_duration_steps} steps)")
        
    def _update_smooth_move(self):
        """Updates the smooth movement."""
        if not self.is_moving:
            return
            
        self.move_progress += 1.0 / self.move_duration_steps
        if self.move_progress >= 1.0:
            self.move_progress = 1.0
            self.is_moving = False
            
        # Linear interpolation
        current_pos = (1.0 - self.move_progress) * self.move_start_pos + self.move_progress * self.move_end_pos
        self.ik_controller.set_target_position(current_pos)
        
    # State update methods
    def _update_idle_state(self):
        """Updates the IDLE state."""
        # Display the menu every 5 seconds
        if self.state_timer % 300 == 1:
            self._show_target_menu()
            
    def _update_approach_state(self):
        """Updates the APPROACH state."""
        if not self.is_moving:
            # Ensure posture correction is disabled (to avoid running during descent)
            self.ik_controller.set_posture_correction_enabled(False)
            # Skip posture adjustment and go directly to descent
            self._transition_to_state(SimpleGraspingState.DESCEND)
            
    def _update_posture_adjust_state(self):
        """Logic for the POSTURE_ADJUST state: skip adjustment and go directly to descent."""
        # Skip posture adjustment and go directly to descent
        print("⏭️ Skipping posture adjustment, proceeding directly to descent.")
        self._transition_to_state(SimpleGraspingState.DESCEND)
        return
            
    def _update_descend_state(self):
        """Updates the DESCEND state - only checks for a green (graspable) state."""
        # Check for a green (graspable) state
        target_path = f"/World/{self.target_prim_name.replace('_object', '')}"
        hit_states = self.pickup_assessor.hit_states.get(target_path, {})
        is_graspable = "green_ray" in hit_states.get("hit_by", set())
        
        if is_graspable:
            print("🟢 Green state detected, ready to grasp!")
            self._transition_to_state(SimpleGraspingState.GRASP)
            return
        
        # Timeout check
        if self.state_timer > self.max_descend_steps:
            print("⏰ Descent timeout. Returning to initial position.")
            self._return_to_initial_position()
            self._transition_to_state(SimpleGraspingState.FAILED)
            return
        
        # Descend step by step
        current_pos = self.ik_controller.current_target_position
        new_pos = current_pos.copy()
        new_pos[2] -= self.descend_step_size
        
        # Safety check - don't descend below the ground
        if new_pos[2] < 0.01:  # At least 1cm from the ground
            print("⚠️ Reached near ground level. Returning to initial position.")
            self._return_to_initial_position()
            self._transition_to_state(SimpleGraspingState.FAILED)
            return
        
        self.ik_controller.set_target_position(new_pos)
        
        # Log status every 30 frames (0.5 seconds)
        if self.state_timer % 30 == 0:
            print(f"⬇️ Descending slowly... Height: {new_pos[2]:.3f}m (searching for green state)")
                
    def _update_grasp_state(self):
        """Updates the GRASP state."""
        # Progressively close the gripper
        if self.state_timer <= self.grasp_duration_steps:
            progress = self.state_timer / self.grasp_duration_steps
            current_pos = (1.0 - progress) * self.grasp_start_pos + progress * self.grasp_end_pos
            self.gripper_controller.target_gripper_position = current_pos
        else:
            self._transition_to_state(SimpleGraspingState.GRASP_SETTLE)
            
    def _update_grasp_settle_state(self):
        """Updates the GRASP_SETTLE state - waits for a short period."""
        if self.state_timer > self.grasp_settle_duration_steps:
            self._transition_to_state(SimpleGraspingState.LIFT)
            
    def _update_lift_state(self):
        """Updates the LIFT state - slowly lifts and checks if the object is following."""
        # Check if the object is following at a set interval
        if self.state_timer % self.lift_check_interval == 0:
            # Check if the grasp was successful
            grasp_success = self.grasp_detector.check_grasp_success()
            
            if not grasp_success:
                print("❌ Grasp failure detected: the object is not being held.")
                print("🔄 Immediately stopping the grasp task and returning to the initial position.")
                self._return_to_initial_position()
                self._transition_to_state(SimpleGraspingState.FAILED)
                return
            
            # Check if the target height has been reached
            current_pos = self.ik_controller.current_target_position
            if current_pos[2] >= self.lift_height:
                print("✅ Reached target lift height with grasp confirmed.")
                self._transition_to_state(SimpleGraspingState.RETREAT)
                return
        
        # Timeout check
        if self.state_timer > 300:  # 5-second timeout
            print("⏰ Lift timeout. The object may not have been grasped. Returning to initial position.")
            self._return_to_initial_position()
            self._transition_to_state(SimpleGraspingState.FAILED)
            return
        
        # Lift step by step
        current_pos = self.ik_controller.current_target_position
        new_pos = current_pos.copy()
        new_pos[2] += self.lift_step_size
        
        # Limit the maximum height
        if new_pos[2] > self.lift_height:
            new_pos[2] = self.lift_height
        
        self.ik_controller.set_target_position(new_pos)
        
        # Log status every 30 frames
        if self.state_timer % 30 == 0:
            print(f"⬆️ Lifting slowly... Height: {new_pos[2]:.3f}m (checking if object is held)")
    
    def _update_retreat_state(self):
        """Updates the RETREAT state - moves to a safe position."""
        # Calculate the safe position and start moving only on the first entry into this state
        if not hasattr(self, '_retreat_safe_position'):
            # Get the object's position at the time of grasp (which remains constant)
            grasp_pos = self.ik_controller.current_target_position.copy()
            grasp_pos[2] = self.lift_height  # Use the lift height
            
            # Calculate the safe position (based on the grasp position, calculated only once)
            self._retreat_safe_position = self.calculate_safe_position(grasp_pos)
            print(f"📍 Retreat safe position calculated (based on grasp position): {self._retreat_safe_position}")
            
            # Get the current position
            current_pos = self.ik_controller.current_target_position
            
            # Check if movement is necessary (if we are already close, transition immediately)
            distance = np.linalg.norm(current_pos - self._retreat_safe_position)
            if distance < 0.01:  # If distance is less than 1cm, consider it reached
                print(f"✅ Already near the safe position. Transitioning directly to transport.")
                self._transition_to_state(SimpleGraspingState.TRANSPORT)
                return
            
            # Start the smooth movement
            self._start_smooth_move(self._retreat_safe_position, self.travel_horizontal_speed)
            print(f"🔄 Moving to safe position...")
            print(f"    Start Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            print(f"    Target Position: [{self._retreat_safe_position[0]:.3f}, {self._retreat_safe_position[1]:.3f}, {self._retreat_safe_position[2]:.3f}]")
        elif not self.is_moving:
            # Movement is complete, transition to the transport state
            print(f"✅ Reached safe position. Transitioning to transport.")
            self._transition_to_state(SimpleGraspingState.TRANSPORT)
            
    def _update_transport_state(self):
        """Updates the TRANSPORT state - moves from the safe position to above the plate."""
        # Check if the object is still held at a regular interval
        if self.state_timer % 30 == 0:
            grasp_success = self.grasp_detector.check_grasp_success()
            if not grasp_success:
                print("❌ Object lost during transport. Grasp failed.")
                print("🔄 Immediately stopping transport and returning to the initial position.")
                self._return_to_initial_position()
                self._transition_to_state(SimpleGraspingState.FAILED)
                return
        
        # Calculate the target position and start moving only on the first entry into this state
        if not hasattr(self, '_transport_final_position'):
            # Use the smart placement manager to calculate the best placement position
            placement_position = self.placement_manager.calculate_placement_position(self.target_prim_name)
            
            if placement_position is not None:
                # Use the calculated placement position
                self._transport_final_position = np.array([placement_position[0], placement_position[1], self.release_height])
                print(f"🎯 Using smart placement position: [{placement_position[0]:.4f}, {placement_position[1]:.4f}, {self.release_height:.4f}]")
            else:
                # Fallback to the plate's center
                if hasattr(self.scene_manager, 'plate_position') and self.scene_manager.plate_position is not None:
                    plate_pos = np.array(self.scene_manager.plate_position)
                    print(f"⚠️ Smart placement failed. Using plate's center position: {self.scene_manager.plate_position}")
                else:
                    plate_pos = np.array([0.28, -0.05, 0.02])
                    print(f"⚠️ Using default plate position: {plate_pos}")
                
                self._transport_final_position = np.array([plate_pos[0], plate_pos[1], self.release_height])
            
            # Get the current position
            current_pos = self.ik_controller.current_target_position
            
            # Check if movement is necessary
            distance = np.linalg.norm(current_pos - self._transport_final_position)
            if distance < 0.01:
                print(f"✅ Already near the plate. Transitioning directly to release.")
                self._transition_to_state(SimpleGraspingState.RELEASE)
                return
            
            # Start the smooth movement
            self._start_smooth_move(self._transport_final_position, self.travel_horizontal_speed)
            print(f"🚚 Transporting to plate: from safe position to directly above the plate.")
            print(f"    Start Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            print(f"    Target Position: [{self._transport_final_position[0]:.3f}, {self._transport_final_position[1]:.3f}, {self._transport_final_position[2]:.3f}]")
        elif not self.is_moving:
            # Movement is complete, transition to the release state
            print(f"✅ Reached plate position. Transitioning to release.")
            self._transition_to_state(SimpleGraspingState.RELEASE)
    
    def _update_release_state(self):
        """Updates the RELEASE state."""
        # Start placement detection after starting the release
        if self.state_timer == 1:
            self.grasp_detector.start_placement_detection()
            print("🔄 Starting placement detection, waiting for the object to stabilize...")
        
        # Continuously check placement status while waiting
        if self.state_timer > 1:
            placement_success = self.grasp_detector.is_object_placed()
            
            if placement_success:
                print("✅ Object successfully placed in the plate.")
                
                # Record the successful placement with the placement manager
                if hasattr(self, '_transport_final_position'):
                    placement_pos = self._transport_final_position.copy()
                    self.placement_manager.record_placement(self.target_prim_name, placement_pos, success=True)
                    print(f"📝 Recorded placement position for {self.target_prim_name}.")
                
                self._transition_to_state(SimpleGraspingState.RETURN_HOME)
                return
        
        # Check if release has timed out
        if self.state_timer > self.release_duration_steps:
            print("❌ Placement detection timed out. Task failed.")
            
            # Record the failed placement
            if hasattr(self, '_transport_final_position'):
                placement_pos = self._transport_final_position.copy()
                self.placement_manager.record_placement(self.target_prim_name, placement_pos, success=False)
                print(f"📝 Recorded failed placement for {self.target_prim_name}.")
            
            self._transition_to_state(SimpleGraspingState.FAILED)
            
    def _update_return_home_state(self):
        """Updates the RETURN_HOME state."""
        # Start moving only on the first entry into this state
        if not hasattr(self, '_return_home_started'):
            # Return to the initial position
            initial_pos = np.array([0.25, 0.0, 0.25])
            current_pos = self.ik_controller.current_target_position
            
            # Check if movement is necessary
            distance = np.linalg.norm(current_pos - initial_pos)
            if distance < 0.01:
                print(f"✅ Already near the initial position. Transitioning directly to success.")
                self._transition_to_state(SimpleGraspingState.SUCCESS)
                return
            
            self._start_smooth_move(initial_pos, self.travel_horizontal_speed)
            print(f"🔙 Returning to initial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
            self._return_home_started = True
        elif not self.is_moving:
            # Return is complete, transition to the success state
            print(f"✅ Returned to initial position. Transitioning to success.")
            self._transition_to_state(SimpleGraspingState.SUCCESS)
    
    def _update_terminal_state(self):
        """Updates a terminal state."""
        # Automatically return to IDLE after 2 seconds
        if self.state_timer > 120:
            self._reset_to_idle()
            
    def _show_target_menu(self):
        """Displays the target selection menu."""
        print("\n" + "="*50)
        print("🤖 Simplified Grasping State Machine - Target Selection")
        print("="*50)
        print("Press a number key to select a target to grasp:")
        
        for i, (prim_path, config) in enumerate(self.target_configs.items()):
            print(f"  [{i+1}] {config['name']}")
            
        print("\nOther Hotkeys:")
        print("  [R]   Reset Scene Positions")
        print("  [V]   Toggle Visualization Display")
        print("  [TAB] Switch Camera")
        print("="*50)
        
    def handle_key_input(self, key):
        """Handles keyboard input."""
        if key == "R":
            # 'R' key: reset the scene and the placement manager
            print("🔄 'R' key pressed: Resetting scene and placement manager.")
            self.reset_scene()
            return True
            
        if self.current_state == SimpleGraspingState.IDLE:
            if key.isdigit():
                success = self.start_grasp_sequence(key)
                if not success:
                    print(f"❌ Failed to start grasp sequence.")
                return success
                
        return False
    
    def _check_plate_stability(self) -> bool:
        """Checks the stability of the plate during the task."""
        if self._monitoring_plate_initial_pos is None or self.plate_name_for_monitoring is None:
            return True # Skip the check if there is no initial position or plate name

        # Get the latest plate object reference from the scene in real-time
        plate_object = self.world.scene.get_object(self.plate_name_for_monitoring)
        if not plate_object or not hasattr(plate_object, 'get_world_pose') or not hasattr(plate_object, 'get_linear_velocity'):
            return True # Skip if the object or its properties can't be accessed

        # Check for position change
        current_pos, _ = plate_object.get_world_pose()
        position_change = np.linalg.norm(np.array(current_pos) - self._monitoring_plate_initial_pos)
        pos_threshold = self.plate_monitoring_config.get('position_threshold', 0.03)
        
        # Check velocity
        current_vel = plate_object.get_linear_velocity()
        speed = np.linalg.norm(np.array(current_vel))
        vel_threshold = self.plate_monitoring_config.get('velocity_threshold', 0.1)

        if position_change > pos_threshold:
            print(f"💥 Significant plate position change detected! Change: {position_change:.3f}m > Threshold: {pos_threshold:.3f}m")
            logger.warning(f"Plate position changed too much ({position_change:.3f}m). Failing task.")
            self.hard_reset_required = True # Request a hard reset
            self.fail_current_task()
            return False

        if speed > vel_threshold:
            print(f"💥 Plate is moving too fast! Speed: {speed:.3f}m/s > Threshold: {vel_threshold:.3f}m/s")
            logger.warning(f"Plate is moving too fast ({speed:.3f}m/s). Failing task.")
            self.hard_reset_required = True # Request a hard reset
            self.fail_current_task()
            return False
            
        return True

    def get_and_clear_hard_reset_flag(self) -> bool:
        """
        Checks if a hard scene reset is required, then clears the flag.
        This is used to handle serious scene errors like the plate being moved.
        """
        reset_needed = self.hard_reset_required
        self.hard_reset_required = False
        return reset_needed

    def reset_scene(self):
        """Resets the scene and the state machine's internal state."""
        if hasattr(self, 'scene_manager') and self.scene_manager:
            self.scene_manager.reset_scene()
        if hasattr(self, 'placement_manager') and self.placement_manager:
            self.placement_manager.reset()
            print("✅ Placement manager has been reset.")
        self._reset_to_idle()
        
    def handle_ik_failure(self):
        """Handles an IK failure by immediately stopping all movement and returning to the initial position."""
        logger.error(f"❌ IK solver failed (current state: {self.current_state.name}), target: {self.ik_controller.current_target_position.tolist()}")
        logger.error("❌ IK solution not found. Immediately stopping current task and returning to initial position.")
        
        # Print failure info at the state machine level
        print("❌ IK unsolvable. Immediately stopping current task and returning to initial position.")
        print(f"🔍 State Machine Failure Info:")
        print(f"   Current State: {self.current_state.get_display_name()}")
        print(f"   Target Object: {self.target_prim_name}")
        print(f"   Is Moving: {self.is_moving}")
        print(f"   Movement Progress: {self.move_progress:.2f}")
        
        # Immediately stop all movement
        self.is_moving = False
        self.move_progress = 0.0
        
        # Immediately return to the initial position
        self._return_to_initial_position()
        
        # Immediately transition to the FAILED state
        self._transition_to_state(SimpleGraspingState.FAILED)
        
    def get_current_state_info(self):
        """Gets information about the current state."""
        return {
            "state": self.current_state,
            "display_name": self.current_state.get_display_name(),
            "timer": self.state_timer,
            "target": self.target_prim_name,
            "is_moving": self.is_moving,
            "frame_count": self.frame_count
        }
    
    def return_to_initial_position(self):
        """Returns to the initial position and resets the state."""
        print("🔄 Returning robot arm to initial position...")
        logger.info("Returning robot arm to initial position due to IK failure.")
        
        # Set the target position to the initial position
        self.ik_controller.set_target_position(self.initial_position)
        
        # Open the gripper
        self.gripper_controller.set_target_position(self.gripper_controller.open_pos)
        
        # Reset the state machine to IDLE
        self._transition_to_state(SimpleGraspingState.IDLE)
    
    def cancel_current_task(self):
        """Cancels the current task."""
        if self.current_state != SimpleGraspingState.IDLE:
            print("❌ Canceling current task...")
            logger.info("User canceled the current task.")
            
            # Clean up task-related state
            self.target_object = None
            self.target_prim_name = None
            self.target_prim_path = None
            
            # Return to the initial state
            self._transition_to_state(SimpleGraspingState.IDLE)
            print("✅ Returned to initial state.")
        
        # Stop all movement
        self.is_moving = False
        
        # Reset the state timer
        self.state_timer = 0
        
        # Clean up fine-tuning state
        if hasattr(self, '_wrist_adjustment_attempts'):
            delattr(self, '_wrist_adjustment_attempts')
    
    def get_placement_summary(self):
        """Gets a summary of placement information."""
        return self.placement_manager.get_placement_summary()
    
    def _scan_existing_oranges(self):
        """Scans for existing oranges in the scene and updates the placement manager."""
        try:
            # Get the names of all orange objects
            orange_names = []
            for target_path, target_config in self.target_configs.items():
                orange_names.append(target_config["name"])
            
            # Scan the scene for oranges
            self.placement_manager.scan_existing_objects(self.world.scene, orange_names)
            
        except Exception as e:
            print(f"❌ Failed to scan for existing oranges: {e}")
            logger.error(f"Failed to scan for existing oranges: {e}")
    
    def _record_data_collection_frame(self):
        """Records a data collection frame."""
        try:
            # Get joint positions (state) - 实际的关节角度
            joint_positions = self.robot.get_joint_positions()[:6]
            
            # Initialize actions with current joint positions as a fallback
            actions = joint_positions.copy()

            # 1. Get action for the arm (joints 0-4) from IK solver
            # This represents the *desired* arm joint angles for the current cartesian target.
            ik_solution, ik_success = self.ik_controller.compute_ik(
                current_joint_positions=joint_positions
            )
            
            if ik_success:
                # If IK solution is found, use it as the action for the arm.
                actions[:5] = self.ik_controller.apply_posture_correction(ik_solution)

            # 2. Get action for the gripper (joint 5) using the hybrid strategy
            gripper_action = 0.0

            # States where the intent is to HOLD the gripper tightly closed
            holding_states = [
                SimpleGraspingState.GRASP_SETTLE,
                SimpleGraspingState.LIFT, 
                SimpleGraspingState.RETREAT,
                SimpleGraspingState.TRANSPORT
            ]

            if self.current_state == SimpleGraspingState.GRASP:
                # STRATEGY 1: For the ACTIVE GRASPING phase, the action is the DYNAMIC, interpolated target.
                # We recalculate the exact same logic from _update_grasp_state here.
                progress = min(1.0, self.state_timer / self.grasp_duration_steps)
                gripper_action = (1.0 - progress) * self.grasp_start_pos + progress * self.grasp_end_pos
            
            elif self.current_state in holding_states:
                # STRATEGY 2: For HOLDING phases, the action is the FINAL intended closed position.
                gripper_action = self.grasp_end_pos
            
            else:
                # STRATEGY 3: For all other states, the action is to be fully OPEN.
                gripper_action = self.gripper_controller.open_pos
            
            actions[5] = gripper_action
            
            # Get camera images (if the camera controller is available)
            front_image = None
            wrist_image = None
            
            
            if self.camera_controller:
                try:
                    # Get front camera image
                    if hasattr(self.camera_controller, 'front_camera') and self.camera_controller.front_camera:
                        front_image = self.camera_controller.front_camera.get_rgba()
                    
                    # Get wrist camera image
                    if hasattr(self.camera_controller, 'wrist_camera') and self.camera_controller.wrist_camera:
                        wrist_image = self.camera_controller.wrist_camera.get_rgba()
                except Exception as e:
                    logger.error(f"❌ Failed to get camera images: {e}")
            
            # Record the frame data
            self.data_collection_manager.record_frame(
                joint_positions=joint_positions,
                actions=actions,
                front_image=front_image,
                wrist_image=wrist_image
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to record data collection frame: {e}")
    
    def _handle_episode_end(self, success: bool):
        """Handles the end of an episode."""
        if self.data_collection_manager:
            # End the data collection episode
            self.data_collection_manager.end_episode(success)
            
            if success:
                print("✅ Episode completed successfully. Data collection has ended.")
                
                # Automatically save successful data in auto mode
                save_result = self.data_collection_manager.save_episode()
                
                if save_result == 'save':
                    print("✅ Data has been successfully saved to the HDF5 file.")
                else:
                    print(f"❌ Failed to save data: {save_result}")
            else:
                print("❌ Episode failed. Data collection has ended.")
                # Do not save failed data
                self.data_collection_manager.discard_episode()

