# -*- coding: utf-8 -*-
"""
Entry point for fully automated data collection.
Modified from the interactive script, this script drives the state machine
by simulating keyboard inputs to achieve automated data collection.
"""

import os
import sys
import time
import logging
import numpy as np

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Logging setup is now handled in src.utils.logger
# Debug print functions are now handled in src.utils.debug_utils
# Config loading functions are now handled in src.utils.config_utils

def main(enable_data_collection=False, auto_mode=False, no_search_mode=False, 
         data_output="./datasets/so101_pickup_auto.hdf5", save_camera_data=False,
         total_success_episodes=10, headless=False):
    """Main function - unified entry point."""
    print("🚀 SO-101 Automated Data Collection System")
    print("=" * 60)
    print("📋 Launch Parameters:")
    print(f"   • enable_data_collection: {enable_data_collection}")
    print(f"   • data_output: {data_output}")
    print(f"   • save_camera_data: {save_camera_data}")
    print(f"   • total_success_episodes: {total_success_episodes}")
    print(f"   • headless: {headless}")
    print("📝 Features include:")
    print("   • Scene loading and object generation")
    print("   • State machine-driven automated grasping loop")
    print("   • Automatic retry for the next target upon failure")
    print("   • Automatic scene reset after completing a round")
    print("   • Automatic exit after reaching the target number of successful grasps")
    print("")
    
    # Load scene configuration
    from src.utils.config_utils import load_scene_config
    scene_config = load_scene_config(PROJECT_ROOT)
    
    # Use the new logging utility module
    from src.utils.logger import setup_logging
    setup_logging()
    
    # 1. Import base modules (before Isaac Sim starts)
    from src.config.config_loader import ConfigLoader
    from src.core.simulation_manager import SimulationManager
    
    try:
        # 2. Load configuration
        print("📋 Step 1: Loading configuration")
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        print("✅ Configuration loaded successfully")
        
        # 3. Start Isaac Sim simulation
        print(f"\n🎮 Step 2: Starting Isaac Sim simulation (headless={headless})")
        sim_manager = SimulationManager(headless=headless)
        simulation_app = sim_manager.start_simulation()
        print("✅ Isaac Sim simulation started")
        
        # 4. Load Isaac Sim extensions and modules
        print("\n🔧 Step 3: Loading Isaac Sim extensions and modules")
        from src.utils.extension_loader import ExtensionLoader
        extension_modules = ExtensionLoader.load_all()
        print("✅ Isaac Sim extensions and modules loaded successfully")
        
        # 5. Import Isaac Sim-related modules (after extensions are loaded)
        from src.core.world_setup import WorldSetup
        from src.robot import get_ik_controller, get_gripper_controller
        from src.input import get_keyboard_handler
        from src.camera import get_multi_camera_controller
        from src.scene.scene_manager import SceneManager
        
        # 6. Create World and scene
        print("\n🌍 Step 4: Creating World and scene")
        world_setup = WorldSetup(config)
        world = world_setup.create_world()
        world_setup.setup_environment()
        world_setup.add_follow_target_task()
        print("✅ World and scene created successfully")
        
        # 7. Load scene objects (oranges and plate)
        print("\n🍊 Step 5: Loading scene objects")
        
        # Use the scene factory to create the orange and plate scene
        from src.utils.scene_factory import SceneFactory
        scene_factory = SceneFactory(PROJECT_ROOT, world)
        scene_objects, orange_positions, plate_center = scene_factory.create_orange_plate_scene(scene_config)
        
        # Record the initial positions of objects for reset
        orange_reset_positions = {}
        
        # Add orange reset positions
        if len(orange_positions) >= 3:
            orange_reset_positions["orange1_object"] = orange_positions[0].tolist()
            orange_reset_positions["orange2_object"] = orange_positions[1].tolist()
            orange_reset_positions["orange3_object"] = orange_positions[2].tolist()
        
        # Add plate reset position
        orange_reset_positions["plate_object"] = plate_center
        
        # Detailed debug output (initial generation)
        from src.utils.debug_utils import print_initial_debug_info
        print_initial_debug_info(plate_center, orange_positions)
        
        # 8. Reset the world and initialize the task
        print("\n🔄 Step 6: Resetting world and initializing task")
        world.reset()
        
        # Wait for task initialization
        print("⏳ Waiting for task initialization...")
        for i in range(60):
            world.step(render=not headless)
            if i % 20 == 0:
                print(f"   Initialization progress: {i+1}/60 steps")
        
        # 9. Get the robot object
        print("\n🤖 Step 7: Acquiring robot object")
        task = world.get_task("so101_follow_target")
        task_params = task.get_params()
        robot_name = task_params["robot_name"]["value"]
        robot = world.scene.get_object(robot_name)
        
        if robot is None:
            raise RuntimeError(f"Robot object not found: {robot_name}")
        print(f"✅ Robot object acquired successfully: {robot.name}")
        
        # 10. Initialize controllers
        print("\n⚙️ Step 8: Initializing controllers")
        
        # IK Controller
        IKController = get_ik_controller()
        ik_controller = IKController(robot, config, PROJECT_ROOT)
        
        # Gripper Controller
        GripperController = get_gripper_controller()
        open_pos = robot.gripper._joint_opened_position
        closed_pos = robot.gripper._joint_closed_position
        gripper_controller = GripperController(open_pos, closed_pos)
        
        # Keyboard Handler
        KeyboardHandler = get_keyboard_handler()
        keyboard_handler = KeyboardHandler(gripper_controller)
        
        # Initialize visualization system
        print("\n🎨 Step 8.5: Initializing visualization system")
        
        from isaacsim.util.debug_draw import _debug_draw
        draw_interface = _debug_draw.acquire_debug_draw_interface()
        print("✅ Debug draw interface acquired")
        
        from src.visualization import (
            get_bbox_visualizer, get_pickup_assessor, 
            get_ray_visualizer, get_debug_visualizer
        )
        
        BoundingBoxVisualizer = get_bbox_visualizer()
        PickupAssessor = get_pickup_assessor()
        RayVisualizer = get_ray_visualizer()
        DebugVisualizer = get_debug_visualizer()
        
        bbox_visualizer = BoundingBoxVisualizer(draw_interface)
        ray_visualizer = RayVisualizer(draw_interface)
        pickup_assessor = PickupAssessor(world, bbox_visualizer)
        debug_visualizer = DebugVisualizer(draw_interface, bbox_visualizer, pickup_assessor, ray_visualizer)
        
        print("✅ Visualization system initialized successfully")
        print("✅ All controllers initialized successfully")
        
        # Initialize camera controller variable (to be created later)
        camera_controller = None
        
        # 11. Scene manager setup
        print("\n🎬 Step 9: Initializing Scene Manager")
        scene_manager = SceneManager(scene_config, world)
        scene_manager.register_scene_objects(scene_objects)
        scene_manager.set_orange_reset_positions(orange_reset_positions)
        print("✅ Scene Manager initialized successfully")
        
        # 12. Create camera controller
        print("\n📷 Step 12: Creating camera controller")
        try:
            from src.camera import get_multi_camera_controller_from_ref
            MultiCameraController = get_multi_camera_controller_from_ref()
            camera_controller = MultiCameraController(config=scene_config)
            print("✅ Camera controller created successfully (parameters read from config)")
        except Exception as e:
            print(f"⚠️ Failed to create camera controller: {e}")
            camera_controller = None
        
        # 13. Connect all systems
        print("\n🔗 Step 13: Connecting all systems")
        if camera_controller:
            keyboard_handler.set_camera_controller(camera_controller)
        keyboard_handler.set_debug_visualizer(debug_visualizer)
        
        try:
            ik_target_sphere = world.scene.get_object("target")
            if ik_target_sphere:
                debug_visualizer.set_ik_target_sphere(ik_target_sphere)
                print(f"✅ IK target sphere connected to visualization system: target")
            else:
                print("⚠️ IK target sphere not found")
        except Exception as e:
            print(f"⚠️ Failed to get IK target sphere: {e}")
        
        # 14. Set base target configurations
        print("\n🔧 Step 14: Setting target configurations")
        from src.utils.config_utils import ConfigManager
        config_manager = ConfigManager(PROJECT_ROOT)
        target_configs = config_manager.get_target_configs(scene_config)
        print("✅ Target configurations loaded from config file")
        
        # 15. Create data collection manager (if enabled)
        data_collection_manager = None
        if enable_data_collection:
            print("\n📊 Step 15: Creating Data Collection Manager")
            output_dir = os.path.dirname(data_output)
            os.makedirs(output_dir, exist_ok=True)
            
            from src.data_collection import DataCollectionManager
            data_collection_manager = DataCollectionManager(
                output_file_path=data_output,
                enable_data_collection=True
            )
            print(f"✅ Data Collection Manager created. Output file: {data_output}")
        else:
            print("\n📊 Step 15: Data collection is disabled")
        
        # 16. Initialize the simplified state machine
        print("\n🤖 Step 16: Initializing simplified state machine")
        from src.state_machine import SimpleGraspingStateMachine
        
        state_machine = SimpleGraspingStateMachine(
            world=world,
            robot=robot,
            ik_controller=ik_controller,
            gripper_controller=gripper_controller,
            pickup_assessor=pickup_assessor,
            scene_manager=scene_manager,
            target_configs=target_configs,
            draw_interface=draw_interface,
            data_collection_manager=data_collection_manager,
            camera_controller=camera_controller
        )
        
        keyboard_handler.set_state_machine(state_machine)
        keyboard_handler.set_scene_manager(scene_manager) # Ensure scene manager is connected
        print("✅ Simplified state machine initialized successfully")

        # Cache Bounding Box info for all objects (critical fix!)
        print("📦 Caching object AABB information...")
        bbox_cache_info = {
            "/World/orange1": "orange1_object",
            "/World/orange2": "orange2_object",
            "/World/orange3": "orange3_object"
        }
        
        for prim_path, prim_name in bbox_cache_info.items():
            bbox_visualizer.cache_prim_extents_and_offset(world, prim_path, prim_name)
        print("✅ AABB information cached successfully")

        # 17. Automated main loop
        print("\n🔄 Step 17: Entering automated data collection loop")
        print("=" * 60)

        successful_grasps = 0
        total_runs = 0

        # Simulate pressing 'V' to ensure debug visualization is off
        print("🙈 Hiding debug visualizations...")
        keyboard_handler.simulate_key_press('v')
        # Step a few frames to ensure it takes effect
        for _ in range(10):
            world.step(render=not headless)
        
        if debug_visualizer.is_enabled:
             print("⚠️ Warning: Could not disable debug visualizations. They may still be visible.")
        else:
             print("✅ Debug visualizations confirmed disabled.")

        start_time = time.time()
        frame_count = 0

        while successful_grasps < total_success_episodes:
            total_runs += 1
            print(f"\n--- 🎬 Starting Run {total_runs} | Successful Grasps: {successful_grasps}/{total_success_episodes} ---")
            
            num_oranges = len(scene_manager.get_oranges())
            
            for i in range(num_oranges):
                target_index = i + 1
                if successful_grasps >= total_success_episodes:
                    break

                print(f"   🍊 Attempting to grasp target: Orange {target_index}")
                
                # Simulate pressing number keys '1', '2', '3' to start grasping
                keyboard_handler.simulate_key_press(str(target_index))
                
                # [Critical Fix] Give the state machine time to react, step a few frames
                # to force it into the is_busy() state.
                print("   ...Waiting for state machine to start...")
                for _ in range(10): # Step 10 frames to ensure state update
                    world.step(render=not headless)
                    frame_count += 1
                    if world.is_playing():
                        # Core update logic, consistent with the main loop
                        current_joint_positions = robot.get_joint_positions()
                        ee_pos, ee_rot = ik_controller.compute_forward_kinematics(frame_name="wrist_link", joint_positions=current_joint_positions[:5])
                        gripper_pos, gripper_rot = ik_controller.compute_forward_kinematics(frame_name="gripper_frame_link", joint_positions=current_joint_positions[:5])
                        ik_data = (ee_pos, ee_rot, gripper_pos, gripper_rot)
                        debug_visualizer.update_calculations(world, ik_data, target_configs, frame_count)
                        state_machine.update()
                        ik_controller.execute_control(robot, state_machine)
                        if not headless:
                            debug_visualizer.draw_visualizations(world, target_configs, frame_count)
                        if camera_controller:
                            camera_controller.update_frame_count()

                # Wait for the state machine to complete the current task
                step_timeout = 60 * 60 # Timeout set to 60 seconds
                step_count = 0
                while state_machine.is_busy() and step_count < step_timeout:
                    world.step(render=not headless)
                    frame_count += 1
                    
                    if world.is_playing():
                        # 1. Calculate IK and FK data
                        current_joint_positions = robot.get_joint_positions()
                        ee_pos, ee_rot = ik_controller.compute_forward_kinematics(frame_name="wrist_link", joint_positions=current_joint_positions[:5])
                        gripper_pos, gripper_rot = ik_controller.compute_forward_kinematics(frame_name="gripper_frame_link", joint_positions=current_joint_positions[:5])
                        ik_data = (ee_pos, ee_rot, gripper_pos, gripper_rot)
                        
                        # 2. Always update visualization calculations
                        debug_visualizer.update_calculations(world, ik_data, target_configs, frame_count)
                        
                        # 3. Update state machine
                        state_machine.update()
                        
                        # 4. Execute IK control
                        ik_controller.execute_control(robot, state_machine)
                        
                        # 5. Draw visualizations if needed
                        if not headless:
                            debug_visualizer.draw_visualizations(world, target_configs, frame_count)
                        
                        # 6. Update cameras
                        if camera_controller:
                            camera_controller.update_frame_count()

                    step_count += 1
                
                if step_count >= step_timeout:
                    print(f"   ⚠️ Timed out while grasping Orange {target_index}. Skipping.")
                    state_machine.fail_current_task() # Handle failure on timeout

                # [New] Check if a hard reset was triggered by plate movement
                if state_machine.get_and_clear_hard_reset_flag():
                    print("💥 Plate movement caused a critical error, triggering scene reset!")
                    keyboard_handler.simulate_key_press('r')
                    # Break out of the inner loop (orange loop) to start the next run
                    break

                # Get grasp result
                was_successful = state_machine.get_last_attempt_status()
                if was_successful:
                    successful_grasps += 1
                    print(f"   ✅ Successfully grasped Orange {target_index}! Total successes: {successful_grasps}")
                else:
                    print(f"   ❌ Failed to grasp Orange {target_index}. Continuing.")

                # [Critical Fix] Ensure state machine has returned to IDLE before proceeding
                print("   ...Waiting for state machine to return to IDLE...")
                idle_wait_timeout = 60 * 5  # 5 second timeout
                idle_step_count = 0
                while state_machine.get_current_state() != "IDLE" and idle_step_count < idle_wait_timeout:
                    world.step(render=not headless)
                    frame_count += 1
                    if world.is_playing():
                        # Core update logic
                        current_joint_positions = robot.get_joint_positions()
                        ee_pos, ee_rot = ik_controller.compute_forward_kinematics(frame_name="wrist_link", joint_positions=current_joint_positions[:5])
                        gripper_pos, gripper_rot = ik_controller.compute_forward_kinematics(frame_name="gripper_frame_link", joint_positions=current_joint_positions[:5])
                        ik_data = (ee_pos, ee_rot, gripper_pos, gripper_rot)
                        debug_visualizer.update_calculations(world, ik_data, target_configs, frame_count)
                        state_machine.update()
                        ik_controller.execute_control(robot, state_machine)
                        if not headless:
                            debug_visualizer.draw_visualizations(world, target_configs, frame_count)
                        if camera_controller:
                            camera_controller.update_frame_count()
                    idle_step_count += 1
                
                if idle_step_count >= idle_wait_timeout:
                    print("   ⚠️ Warning: Timed out waiting for state machine to return to IDLE! Problems may occur.")
            
            if successful_grasps >= total_success_episodes:
                break

            # All oranges in the current run have been attempted, simulate 'R' key press to reset scene
            print("   🔄 All targets attempted. Resetting scene...")
            keyboard_handler.simulate_key_press('r')
            
            # Wait for scene reset to complete
            step_timeout = 60 * 5 # 5 second timeout
            step_count = 0
            # Wait for state machine to return to the initial IDLE state
            while state_machine.get_current_state() != "IDLE" and step_count < step_timeout:
                world.step(render=not headless)
                frame_count += 1
                
                if world.is_playing():
                    # 1. Calculate IK and FK data
                    current_joint_positions = robot.get_joint_positions()
                    ee_pos, ee_rot = ik_controller.compute_forward_kinematics(frame_name="wrist_link", joint_positions=current_joint_positions[:5])
                    gripper_pos, gripper_rot = ik_controller.compute_forward_kinematics(frame_name="gripper_frame_link", joint_positions=current_joint_positions[:5])
                    ik_data = (ee_pos, ee_rot, gripper_pos, gripper_rot)

                    # 2. Always update visualization calculations
                    debug_visualizer.update_calculations(world, ik_data, target_configs, frame_count)
                    
                    # 3. Update state machine
                    state_machine.update()

                    # 4. Execute IK control
                    ik_controller.execute_control(robot, state_machine)
                    
                    # 5. Draw visualizations if needed
                    if not headless:
                        debug_visualizer.draw_visualizations(world, target_configs, frame_count)

                    # 6. Update cameras
                    if camera_controller:
                        camera_controller.update_frame_count()

                step_count += 1

        end_time = time.time()
        print(f"\n🎉 Task completed! Total successful grasps: {successful_grasps}.")
        print(f"   Total time: {end_time - start_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n⌨️ Keyboard interrupt received, exiting...")
    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up resources
        print("\n🧹 Cleaning up resources...")
        try:
            if data_collection_manager:
                data_collection_manager.close() # Ensure data is saved
            if 'simulation_app' in locals():
                simulation_app.close()
        except Exception as e:
            print(f"⚠️ An error occurred during resource cleanup: {e}")
    
    print("✅ Program exited normally")
    return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fully automated data collection script.")
    parser.add_argument("--enable-data-collection", action="store_true", default=True, help="Enable data collection (enabled by default).")
    parser.add_argument("--data-output", type=str, default="./datasets/automatic_collection.hdf5", help="Output path for the data file.")
    parser.add_argument("--save-camera-data", action="store_true", help="Enable saving of camera data.")
    parser.add_argument("--total-success-episodes", type=int, default=10, help="Target number of successful grasps.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    
    args, unknown_args = parser.parse_known_args()
    
    # Pass arguments to the main function
    exit(main(enable_data_collection=args.enable_data_collection,
              data_output=args.data_output,
              save_camera_data=args.save_camera_data,
              total_success_episodes=args.total_success_episodes,
              headless=args.headless))
