# -*- coding: utf-8 -*-
"""
Interactive Data Collection Entry Point
Integrates scene loading, IK control, keyboard interaction, camera control, and more.
This is the unified entry point for both data collection and VLA model testing.
"""

import os
import sys
import time
import logging
import numpy as np
import cv2

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Logging setup has been moved to src.utils.logger
# Debug print functions have been moved to src.utils.debug_utils
# Configuration loading functions have been moved to src.utils.config_utils

def display_camera_images(front_image, wrist_image, frame_count, joint_positions, current_camera_info=None):
    """Displays camera images and status information using OpenCV."""
    try:
        if front_image is not None and wrist_image is not None:
            # Ensure images are 3-channel and convert color format (RGB -> BGR)
            if len(front_image.shape) == 2:
                front_image_bgr = cv2.cvtColor(front_image, cv2.COLOR_GRAY2BGR)
            else:
                front_image_bgr = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
                
            if len(wrist_image.shape) == 2:
                wrist_image_bgr = cv2.cvtColor(wrist_image, cv2.COLOR_GRAY2BGR)
            else:
                wrist_image_bgr = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)

            # Determine original image dimensions
            front_h, front_w, _ = front_image_bgr.shape
            wrist_h, wrist_w, _ = wrist_image_bgr.shape
            
            # Create a status information image with a height matching the tallest camera image
            status_h = max(front_h, wrist_h)
            status_w = 400  # Width of the status panel
            status_img = np.zeros((status_h, status_w, 3), dtype=np.uint8)
            
            # Add text information
            cv2.putText(status_img, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(status_img, "Joint Positions:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display joint positions
            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
            for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
                y_pos = 100 + i * 25
                cv2.putText(status_img, f"{name}: {pos:.3f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display current camera information
            if current_camera_info:
                camera_name = current_camera_info.get("name", "Unknown")
                cv2.putText(status_img, f"Camera: {camera_name}", (10, status_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Add control hints
            cv2.putText(status_img, "Controls:", (10, status_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(status_img, "TAB: Switch Camera", (10, status_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # If image heights differ, pad the shorter image to match the taller one
            if front_h != wrist_h:
                if front_h > wrist_h:
                    padding = np.zeros((front_h - wrist_h, wrist_w, 3), dtype=np.uint8)
                    wrist_image_bgr = np.vstack([wrist_image_bgr, padding])
                else:
                    padding = np.zeros((wrist_h - front_h, front_w, 3), dtype=np.uint8)
                    front_image_bgr = np.vstack([front_image_bgr, padding])

            # Horizontally stack images: two camera views + status information
            display_img = np.hstack([front_image_bgr, wrist_image_bgr, status_img])
            
            # Display the image
            cv2.imshow('SO-101 Data Collection - Camera Views & Status', display_img)
            cv2.waitKey(1)  # Non-blocking wait
            
    except Exception as e:
        print(f"⚠️ Failed to display images: {e}")

def get_camera_images_for_display(camera_controller):
    """Gets camera images for display with OpenCV."""
    try:
        front_image = None
        wrist_image = None
        
        if camera_controller:
            try:
                # Get front camera image
                if hasattr(camera_controller, 'front_camera') and camera_controller.front_camera:
                    front_rgba = camera_controller.front_camera.get_rgba()
                    if front_rgba is not None and len(front_rgba.shape) == 3:
                        # Convert to RGB and ensure correct data type
                        front_image = front_rgba[:, :, :3].astype(np.uint8)
                        # Ensure the image has 3 channels
                        if front_image.shape[2] != 3:
                            print(f"⚠️ Front camera image has an unexpected number of channels: {front_image.shape}")
                
                # Get wrist camera image
                if hasattr(camera_controller, 'wrist_camera') and camera_controller.wrist_camera:
                    wrist_rgba = camera_controller.wrist_camera.get_rgba()
                    if wrist_rgba is not None and len(wrist_rgba.shape) == 3:
                        # Convert to RGB and ensure correct data type
                        wrist_image = wrist_rgba[:, :, :3].astype(np.uint8)
                        # Ensure the image has 3 channels
                        if wrist_image.shape[2] != 3:
                            print(f"⚠️ Wrist camera image has an unexpected number of channels: {wrist_image.shape}")
            except Exception as e:
                print(f"⚠️ Failed to get camera images: {e}")
                import traceback
                traceback.print_exc()
        
        return front_image, wrist_image
        
    except Exception as e:
        print(f"❌ Failed to get camera images: {e}")
        return None, None

def main(enable_data_collection=False, auto_mode=False, no_search_mode=False, 
         data_output="./datasets/so101_pickup_data.hdf5", save_camera_data=False, 
         enable_opencv_display=False):
    """Main function - the unified entry point."""
    print("🚀 SO-101 Interactive Data Collection System")
    print("=" * 60)
    print("📋 Startup Parameters:")
    print(f"   - enable_data_collection: {enable_data_collection}")
    print(f"   - auto_mode: {auto_mode}")
    print(f"   - no_search_mode: {no_search_mode}")
    print(f"   - data_output: {data_output}")
    print(f"   - save_camera_data: {save_camera_data}")
    print(f"   - enable_opencv_display: {enable_opencv_display}")
    print("   - enable_cameras: Handled automatically by Isaac Sim")
    print("📝 Features:")
    print("   - Scene loading and object generation")
    print("   - IK control and gripper operation") 
    print("   - Keyboard controls ('R' to reset, 'TAB' to switch camera)")
    print("   - Multi-camera view switching")
    print("   - Smart object placement to avoid overlaps")
    if enable_opencv_display:
        print("   - Real-time camera display with OpenCV")
    print("")
    
    # Load configuration file
    from src.utils.config_utils import load_scene_config
    scene_config = load_scene_config(PROJECT_ROOT)
    
    # Use the new logging utility module
    from src.utils.logger import setup_logging
    setup_logging()
    
    # 1. Import base modules (before starting Isaac Sim)
    from src.config.config_loader import ConfigLoader
    from src.core.simulation_manager import SimulationManager
    
    try:
        # 2. Load configuration
        print("📋 Step 1: Loading configuration")
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        print("✅ Configuration loaded.")
        
        # 3. Start Isaac Sim simulation
        print("\n🎮 Step 2: Starting Isaac Sim")
        sim_manager = SimulationManager(headless=False)  # Visualization mode
        simulation_app = sim_manager.start_simulation()
        print("✅ Isaac Sim started.")
        
        # 4. Load Isaac Sim extensions and modules
        print("\n🔧 Step 3: Loading Isaac Sim extensions and modules")
        from src.utils.extension_loader import ExtensionLoader
        extension_modules = ExtensionLoader.load_all()
        print("✅ Isaac Sim extensions and modules loaded.")
        
        # 5. Import Isaac Sim-dependent modules (after loading extensions)
        from src.core.world_setup import WorldSetup
        from src.robot import get_ik_controller, get_gripper_controller
        from src.input import get_keyboard_handler, get_input_manager
        from src.camera import get_multi_camera_controller
        from src.scene.scene_manager import SceneManager
        
        # 6. Create World and scene
        print("\n🌍 Step 4: Creating World and scene")
        world_setup = WorldSetup(config)
        world = world_setup.create_world()
        world_setup.setup_environment()
        world_setup.add_follow_target_task()
        print("✅ World and scene created.")
        
        # 7. Load scene objects (oranges and plate)
        print("\n🍊 Step 5: Loading scene objects")
        
        # Use the SceneFactory to create the orange and plate scene
        from src.utils.scene_factory import SceneFactory
        scene_factory = SceneFactory(PROJECT_ROOT, world)
        scene_objects, orange_positions, plate_center = scene_factory.create_orange_plate_scene(scene_config)
        
        # Record the initial positions of objects for resets
        orange_reset_positions = {}
        
        # Add orange reset positions
        if len(orange_positions) >= 3:
            orange_reset_positions["orange1_object"] = orange_positions[0].tolist()
            orange_reset_positions["orange2_object"] = orange_positions[1].tolist()
            orange_reset_positions["orange3_object"] = orange_positions[2].tolist()
        
        # Add plate reset position
        orange_reset_positions["plate_object"] = plate_center
        
        # Set orange1, orange2, orange3 variables for compatibility
        orange1 = scene_objects.get("orange1_object")
        orange2 = scene_objects.get("orange2_object") 
        orange3 = scene_objects.get("orange3_object")
        
        # Detailed debug output (for initial generation)
        from src.utils.debug_utils import print_initial_debug_info
        print_initial_debug_info(plate_center, orange_positions)
        
        # 8. Reset world and initialize task
        print("\n🔄 Step 6: Resetting world and initializing task")
        world.reset()
        
        # Wait for the task to initialize
        print("⏳ Waiting for task to initialize...")
        for i in range(60):
            world.step(render=True)
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
        print(f"✅ Robot object acquired: {robot.name}")
        
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
        
        # Input Manager
        InputManager = get_input_manager()
        input_manager = InputManager()
        
        # Initialize visualization system
        print("\n🎨 Step 8.5: Initializing visualization system")
        
        # Import visualization modules
        from isaacsim.util.debug_draw import _debug_draw
        draw_interface = _debug_draw.acquire_debug_draw_interface()
        print("✅ Debug draw interface acquired.")
        
        # Create visualization components
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
        
        print("✅ Visualization system initialized.")
        
        print("✅ All controllers initialized.")
        
        # Initialize camera controller variable (to be created later)
        camera_controller = None
        
        # 11. Set up Scene Manager
        print("\n🎬 Step 9: Initializing Scene Manager")
        scene_manager = SceneManager(scene_config, world)
        
        # Register scene objects (using the ones that were actually loaded)
        scene_manager.register_scene_objects(scene_objects)
        
        # Set orange reset positions
        scene_manager.set_orange_reset_positions(orange_reset_positions)
        
        print("✅ Scene Manager initialized.")
        
        # 12. Create Camera Controller (create early to ensure it's available for the InputManager)
        print("\n📷 Step 12: Creating Camera Controller")
        
        try:
            from src.camera import get_multi_camera_controller_from_ref
            MultiCameraController = get_multi_camera_controller_from_ref()
            
            # Create the camera controller (reading parameters from config)
            camera_controller = MultiCameraController(config=scene_config)
            print("✅ Camera Controller created (reading parameters from config).")
        except Exception as e:
            print(f"⚠️ Failed to create Camera Controller: {e}")
            camera_controller = None
        
        # 13. Connect all systems
        print("\n🔗 Step 13: Connecting all systems")
        
        # Connect the keyboard handler to various components
        if camera_controller:
            keyboard_handler.set_camera_controller(camera_controller)
        keyboard_handler.set_debug_visualizer(debug_visualizer)
        
        # Set the IK target sphere for the debug visualizer
        try:
            ik_target_sphere = world.scene.get_object("target")
            if ik_target_sphere:
                debug_visualizer.set_ik_target_sphere(ik_target_sphere)
                print(f"✅ IK target sphere connected to visualization system: target")
            else:
                print("⚠️ IK target sphere not found. 'V' key will not control its visibility.")
        except Exception as e:
            print(f"⚠️ Failed to get IK target sphere: {e}")
        
        # 14. Set up base target configurations
        print("\n🔧 Step 14: Setting up target configurations")
        from src.utils.config_utils import ConfigManager
        config_manager = ConfigManager(PROJECT_ROOT)
        target_configs = config_manager.get_target_configs(scene_config)
        print("✅ Target configurations loaded from config file.")
        
        # 15. Create Data Collection Manager (if enabled)
        data_collection_manager = None
        if enable_data_collection:
            print("\n📊 Step 15: Creating Data Collection Manager")
            # Ensure the output directory exists
            output_dir = os.path.dirname(data_output)
            os.makedirs(output_dir, exist_ok=True)
            
            from src.data_collection import DataCollectionManager
            data_collection_manager = DataCollectionManager(
                output_file_path=data_output,
                enable_data_collection=True
            )
            print(f"✅ Data Collection Manager created. Output file: {data_output}")
        else:
            print("\n📊 Step 15: Data collection is disabled.")
        
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
        
        # Connect the state machine to the keyboard handler
        keyboard_handler.set_state_machine(state_machine)
        print("✅ Simplified state machine initialized.")
        
        # Cache bounding box information for all objects
        print("📦 Caching object bounding box information...")
        bbox_cache_info = {
            "/World/orange1": "orange1_object",
            "/World/orange2": "orange2_object", 
            "/World/orange3": "orange3_object"
        }
        
        for prim_path, prim_name in bbox_cache_info.items():
            bbox_visualizer.cache_prim_extents_and_offset(world, prim_path, prim_name)
        
        
        input_manager.setup(
            keyboard_handler=keyboard_handler,
            camera_controller=camera_controller, 
            scene_manager=scene_manager,
            ik_controller=ik_controller
        )
        print("✅ All systems connected.")
        
        # 16.5. Initialize OpenCV display (if enabled)
        opencv_gui_available = False
        if enable_opencv_display:
            try:
                # Check if OpenCV has GUI capabilities
                if hasattr(cv2, 'namedWindow') and hasattr(cv2, 'WINDOW_AUTOSIZE'):
                    cv2.namedWindow('SO-101 Data Collection - Camera Views & Status', cv2.WINDOW_AUTOSIZE)
                    opencv_gui_available = True
                    print("✅ OpenCV visualization window created.")
                else:
                    print("⚠️ OpenCV GUI functions are not available (missing GUI module).")
                    print("📊 Will use console output to display data information.")
                    opencv_gui_available = False
            except Exception as e:
                print(f"⚠️ OpenCV GUI is not available: {e}")
                print("📊 Will use console output to display data information.")
                opencv_gui_available = False
        
        # 17. Display control hints
        print("\n🎹 Keyboard Controls:")
        print("   'R'   - Reset the scene (objects are randomly repositioned, robot returns to initial pose)")
        print("   'TAB' - Switch camera view (main/wrist/front)")
        print("   'V'   - Toggle debug visualizations (bounding boxes, rays, IK target)")
        print("   'Q'   - Quit the program")
        print("   'C'   - Cancel the current operation")
        if enable_opencv_display and opencv_gui_available:
            print("   📷 OpenCV Window - Displays dual camera feeds and status in real-time")
        print("")
        
        # 18. Main loop
        print("🎮 Entering interactive mode. Keyboard controls are active...")
        print("=" * 60)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Process user input
            input_manager.process_input()
            
            # Check for exit condition
            if keyboard_handler.peek_user_choice() == "q":
                print("\n👋 Exit command received. Shutting down...")
                break
            
            # Simulation step
            world.step(render=True)
            frame_count += 1
            
            # Update visualizations (every frame)
            if world.is_playing():
                try:
                    # Calculate IK and FK data
                    current_joint_positions = robot.get_joint_positions()
                    
                    # Compute end-effector and gripper poses
                    ee_pos, ee_rot = ik_controller.compute_forward_kinematics(
                        frame_name="wrist_link", joint_positions=current_joint_positions[:5]
                    )
                    gripper_pos, gripper_rot = ik_controller.compute_forward_kinematics(
                        frame_name="gripper_frame_link", joint_positions=current_joint_positions[:5]
                    )
                    
                    # Pack IK data
                    ik_data = (ee_pos, ee_rot, gripper_pos, gripper_rot)
                    
                    # Update visualization system (new two-step process)
                    # 1. Always perform mathematical calculations to ensure logic like raycasting is active
                    debug_visualizer.update_calculations(world, ik_data, target_configs, frame_count)
                    
                    # 2. Perform on-screen drawing based on the 'V' key state
                    debug_visualizer.draw_visualizations(world, target_configs, frame_count)
                    
                    # Update the state machine
                    state_machine.update()
                    
                    # Update camera frame count (if camera controller exists)
                    if camera_controller:
                        camera_controller.update_frame_count()
                    
                    # OpenCV display (if enabled)
                    if enable_opencv_display and opencv_gui_available:
                        try:
                            # Get camera images
                            front_img, wrist_img = get_camera_images_for_display(camera_controller)
                            
                            # Get current camera info
                            current_camera_info = None
                            if camera_controller:
                                current_camera_info = camera_controller.get_current_camera_info()
                            
                            # Display images
                            if front_img is not None and wrist_img is not None:
                                display_camera_images(front_img, wrist_img, frame_count, current_joint_positions, current_camera_info)
                        except Exception as e:
                            if frame_count % 300 == 0:  # Avoid spamming errors
                                print(f"⚠️ Failed to display with OpenCV: {e}")
                    
                    # Execute robot arm control (critical step) - pass the state machine
                    success = ik_controller.execute_control(robot, state_machine)
                    
                except Exception as e:
                    if frame_count % 300 == 0:  # Avoid spamming errors
                        print(f"⚠️ Failed to update visualization: {e}")
            
            # Display status information every 5 seconds
            if frame_count % 300 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                current_camera = camera_controller.get_current_camera_info()
                camera_name = current_camera["name"] if current_camera else "Unknown"
                
                print(f"🎮 Status: {frame_count:6d} frames | {fps:5.1f} FPS | Current Camera: {camera_name}")
        
    except KeyboardInterrupt:
        print("\n⌨️ Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up resources
        print("\n🧹 Cleaning up resources...")
        try:
            # Clean up OpenCV window
            if enable_opencv_display and 'opencv_gui_available' in locals() and opencv_gui_available:
                cv2.destroyAllWindows()
                print("✅ OpenCV window closed.")
            
            if 'input_manager' in locals():
                input_manager.cleanup()
            if 'simulation_app' in locals():
                simulation_app.close()
        except Exception as e:
            print(f"⚠️ An error occurred during resource cleanup: {e}")
    
    print("✅ Program exited normally.")
    return 0

if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments (removed --enable_cameras, as it's handled by Isaac Sim)
    parser = argparse.ArgumentParser(description="Interactive data collection script.")
    parser.add_argument("--enable-data-collection", action="store_true", help="Enable data collection.")
    parser.add_argument("--auto", action="store_true", help="Enable automatic mode.")
    parser.add_argument("--no-search-mode", action="store_true", help="Disable search mode.")
    parser.add_argument("--data-output", type=str, default="./datasets/so101_pickup_data.hdf5", help="Path for the data output file.")
    parser.add_argument("--save-camera-data", action="store_true", help="Enable saving of camera data.")
    parser.add_argument("--enable-opencv-display", action="store_true", help="Enable real-time camera display with OpenCV.")
    
    args, unknown_args = parser.parse_known_args()
    
    # Pass arguments to the main function
    exit(main(enable_data_collection=args.enable_data_collection,
              auto_mode=args.auto,
              no_search_mode=args.no_search_mode,
              data_output=args.data_output,
              save_camera_data=args.save_camera_data,
              enable_opencv_display=args.enable_opencv_display))
