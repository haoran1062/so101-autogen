# -*- coding: utf-8 -*-
"""
VLA (Vision-Language-Action) Inference Script.
"""

import sys
import os
import logging
import numpy as np
import time
import argparse
import cv2
import torch
import csv
from datetime import datetime
import threading
from queue import Queue

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz

    def sleep(self):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        
        sleep_time = next_wakeup_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        self.last_time += self.sleep_duration

        if time.time() > self.last_time + self.sleep_duration:
            self.last_time = time.time()

class ActionHistoryRecorder:
    """Records the history of actions taken."""
    
    def __init__(self, output_dir="./action_history"):
        """Initializes the recorder."""
        self.output_dir = output_dir
        self.csv_file = None
        self.csv_writer = None
        self.step_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"action_history_{timestamp}.csv")
        
        self.setup_csv()
    
    def setup_csv(self):
        """Sets up the CSV file."""
        try:
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            
            # Define CSV column headers
            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
            headers = ['timestamp', 'step', 'round']
            
            # Add columns for joint positions and actions
            for joint in joint_names:
                headers.extend([f'{joint}_position', f'{joint}_action'])
            
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(headers)
            
            logger.info(f"📊 Action history recorder initialized: {self.csv_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to set up CSV file: {e}")
            raise
    
    def record_action(self, step, round_idx, joint_positions, actions):
        """Records action data."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Prepare data row
            row = [timestamp, step, round_idx]
            
            # Add joint position and action data
            for i in range(6):  # 6 joints
                joint_pos = joint_positions[i] if i < len(joint_positions) else 0.0
                action = actions[i] if i < len(actions) else 0.0
                row.extend([joint_pos, action])
            
            # Write to CSV
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # Immediately write to disk
            
            self.step_count += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to record action data: {e}")
    
    def close(self):
        """Closes the recorder."""
        try:
            if self.csv_file:
                self.csv_file.close()
                logger.info(f"📊 Action history recording complete. Total steps recorded: {self.step_count}. File: {self.csv_path}")
        except Exception as e:
            logger.error(f"❌ Failed to close recorder: {e}")

def display_camera_images(front_image, wrist_image, step, joint_positions, action_data=None, vla_input_data=None):
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
            
            # Create a status information image, with height matching the taller camera image
            status_h = max(front_h, wrist_h)
            status_w = 400  # Width of the status panel
            status_img = np.zeros((status_h, status_w, 3), dtype=np.uint8)
            
            # Add text information
            cv2.putText(status_img, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(status_img, "Joint Positions:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display joint positions
            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
            for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
                y_pos = 100 + i * 25
                cv2.putText(status_img, f"{name}: {pos:.3f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # If action data is available, display action info
            if action_data is not None:
                cv2.putText(status_img, f"Action Shape: {action_data.shape}", (10, status_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # If image heights differ, pad the shorter image to match the taller one
            if front_h != wrist_h:
                if front_h > wrist_h:
                    padding = np.zeros((front_h - wrist_h, wrist_w, 3), dtype=np.uint8)
                    wrist_image_bgr = np.vstack([wrist_image_bgr, padding])
                else:
                    padding = np.zeros((wrist_h - front_h, front_w, 3), dtype=np.uint8)
                    front_image_bgr = np.vstack([front_image_bgr, padding])

            # Horizontally stack images: two camera views + status info
            display_img = np.hstack([front_image_bgr, wrist_image_bgr, status_img])
            
            # Display the image
            cv2.imshow('VLA Test - Camera Views & Status', display_img)
            cv2.waitKey(1)  # Non-blocking wait
            
    except Exception as e:
        print(f"⚠️ Failed to display images: {e}")

def get_observation_with_visualization(world, robot, camera_controller, task_description="pick and place the orange"):
    """Gets observation data and returns raw images for visualization."""
    try:
        # Get joint positions
        joint_positions = robot.get_joint_positions()[:6]  # Only take the first 6 joints
        
        # Get camera images
        front_image = None
        wrist_image = None
        
        if camera_controller:
            try:
                # Get front camera image
                if hasattr(camera_controller, 'front_camera') and camera_controller.front_camera:
                    front_rgba = camera_controller.front_camera.get_rgba()
                    if front_rgba is not None and len(front_rgba.shape) == 3:
                        # Convert to RGB format, ensure correct data type
                        front_image = front_rgba[:, :, :3].astype(np.uint8)
                        # Ensure the image is 3-channel
                        if front_image.shape[2] != 3:
                            print(f"⚠️ Abnormal channel count for front camera image: {front_image.shape}")
                
                # Get wrist camera image
                if hasattr(camera_controller, 'wrist_camera') and camera_controller.wrist_camera:
                    wrist_rgba = camera_controller.wrist_camera.get_rgba()
                    if wrist_rgba is not None and len(wrist_rgba.shape) == 3:
                        # Convert to RGB format, ensure correct data type
                        wrist_image = wrist_rgba[:, :, :3].astype(np.uint8)
                        # Ensure the image is 3-channel
                        if wrist_image.shape[2] != 3:
                            print(f"⚠️ Abnormal channel count for wrist camera image: {wrist_image.shape}")
            except Exception as e:
                print(f"⚠️ Failed to get camera images: {e}")
                import traceback
                traceback.print_exc()
        
        # Build observation data
        if front_image is not None and wrist_image is not None:
            # Ensure image dimensions are consistent
            if front_image.shape != wrist_image.shape:
                print(f"⚠️ Camera image dimensions do not match - Front: {front_image.shape}, Wrist: {wrist_image.shape}")
                # Resize to a consistent shape
                target_shape = (480, 640, 3)
                front_image = cv2.resize(front_image, (target_shape[1], target_shape[0]))
                wrist_image = cv2.resize(wrist_image, (target_shape[1], target_shape[0]))
            
            observation = {
                "front": torch.from_numpy(front_image[None, ...]),
                "wrist": torch.from_numpy(wrist_image[None, ...]),
                "joint_pos": torch.from_numpy(joint_positions[None, ...]),
                "task_description": task_description
            }
        else:
            # Use default images if acquisition fails
            observation = {
                "front": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
                "wrist": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
                "joint_pos": torch.from_numpy(joint_positions[None, ...]),
                "task_description": task_description
            }
        
        return observation, front_image, wrist_image, joint_positions
        
    except Exception as e:
        print(f"❌ Failed to get observation data: {e}")
        # Return default observation data on failure
        return {
            "front": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
            "wrist": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
            "joint_pos": torch.zeros(1, 6, dtype=torch.float32),
            "task_description": task_description
        }, None, None, None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fixed VLA (Vision-Language-Action) Inference Script")
    parser.add_argument("--policy_host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=4399, help="Policy server port")
    parser.add_argument("--policy_checkpoint_path", type=str, 
                       default="/home/haoran/projects/lerobot/outputs/train/my_smolvla/checkpoints/last/pretrained_model",
                       help="Path to the policy model")
    parser.add_argument("--policy_action_horizon", type=int, default=50, help="Action sequence length (horizon)")
    parser.add_argument("--policy_language_instruction", type=str, default="pick and place the orange", help="Language instruction for the policy")
    parser.add_argument("--episode_length_s", type=float, default=90.0, help="Episode length in seconds")
    parser.add_argument("--eval_rounds", type=int, default=1, help="Number of evaluation rounds")
    parser.add_argument("--step_hz", type=float, default=60.0, help="Simulation frequency (Hz)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    
    args = parser.parse_args()
    
    print("🚀 VLA Inference Script (Fixed)")
    print("=" * 50)
    print(f"📋 Test Parameters:")
    print(f"   Server: {args.policy_host}:{args.policy_port}")
    print(f"   Model Path: {args.policy_checkpoint_path}")
    print(f"   Action Horizon: {args.policy_action_horizon}")
    print(f"   Language Instruction: {args.policy_language_instruction}")
    print(f"   Episode Length: {args.episode_length_s}s")
    print(f"   Evaluation Rounds: {args.eval_rounds}")
    print(f"   Simulation Frequency: {args.step_hz}Hz")
    print(f"   Headless Mode: {args.headless}")
    
    try:
        # Import VLA-related modules
        from src.vla.vla_policy_client import SO101VLAPolicyClient
        from src.vla.environment_adapter import IsaacSimEnvironmentAdapter
        
        # Create VLA client
        print("🔗 Creating VLA client...")
        vla_client = SO101VLAPolicyClient(
            host=args.policy_host,
            port=args.policy_port,
            checkpoint_path=args.policy_checkpoint_path,
            action_horizon=args.policy_action_horizon,
            language_instruction=args.policy_language_instruction
        )
        print("✅ VLA client created successfully")
        
        # Create action history recorder
        print("📊 Initializing action history recorder...")
        action_recorder = ActionHistoryRecorder()
        print("✅ Action history recorder created successfully")
        
        # Action sequence management (following the reference script's approach)
        actions = None
        current_action_index = 0
        
        # Initialize the real environment (using the data collection script's method)
        print("🔧 Initializing the real environment...")
        
        # Import environment initialization module
        from src.vla.environment_initializer import initialize_environment_for_vla
        
        # Initialize the real environment
        world, robot, camera_controller, config = initialize_environment_for_vla(PROJECT_ROOT, headless=args.headless)
        
        # After environment initialization, read debug visualization settings from config and hide elements
        print("🔧 Reading debug visualization configuration...")
        try:
            from src.config.config_loader import ConfigLoader
            config_loader = ConfigLoader()
            args_cli = config_loader.get_args_cli()
            show_debug_viz = args_cli.show_debug_viz
            print(f"📋 Debug visualization setting: {show_debug_viz}")
            
            # If set to false in config, hide debug visualization elements
            if not show_debug_viz:
                print("🔧 Hiding debug visualization elements (IK target, rays, etc.)...")
                
                # Use the same visualization system as the data collection script
                from isaacsim.util.debug_draw import _debug_draw
                draw_interface = _debug_draw.acquire_debug_draw_interface()
                
                # Create visualization components (same as data collection script)
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
                
                # Set the IK target sphere for the debug visualizer (same as data collection script)
                try:
                    ik_target_sphere = world.scene.get_object("target")
                    if ik_target_sphere:
                        debug_visualizer.set_ik_target_sphere(ik_target_sphere)
                        print(f"✅ IK target sphere connected to visualization system: target")
                    else:
                        print("⚠️ IK target sphere not found")
                except Exception as e:
                    print(f"⚠️ Failed to get IK target sphere: {e}")
                
                # Forcefully hide all debug visualizations
                debug_visualizer.is_enabled = False
                # Manually hide the IK target sphere
                debug_visualizer._toggle_ik_target_sphere()
                print("✅ Debug visualization elements hidden")
            else:
                print("ℹ️ Debug visualization is enabled and will remain visible.")
                debug_visualizer = None
        except Exception as e:
            print(f"⚠️ Failed to process debug visualization configuration: {e}")
            debug_visualizer = None
        
        # Save debug visualization setting for later cleanup
        debug_clear_enabled = not show_debug_viz
        
        # Verify environment initialization
        if robot is None:
            raise RuntimeError("❌ Robot object initialization failed")
        if camera_controller is None:
            raise RuntimeError("❌ Camera controller initialization failed")
        
        print(f"✅ Robot object: {robot.name}")
        print(f"✅ Camera controller: {type(camera_controller).__name__}")
        
        # Create environment adapter
        env_adapter = IsaacSimEnvironmentAdapter(world, robot, camera_controller)
        print("✅ Environment adapter created successfully")
        
        # Initialize Scene Manager to handle reset logic
        print("🎬 Initializing Scene Manager...")
        from src.scene.scene_manager import SceneManager
        from src.utils.config_utils import load_scene_config
        
        scene_config = load_scene_config(PROJECT_ROOT)
        scene_manager = SceneManager(scene_config, world)
        
        # Register scene objects for reset
        try:
            scene_objects = {
                "orange1_object": world.scene.get_object("orange1_object"),
                "orange2_object": world.scene.get_object("orange2_object"),
                "orange3_object": world.scene.get_object("orange3_object"),
                "plate_object": world.scene.get_object("plate_object")
            }
            scene_manager.register_scene_objects(scene_objects)
            print("✅ Scene objects registered with manager")
        except Exception as e:
            print(f"⚠️ Failed to register scene objects, reset functionality may be limited: {e}")
            scene_manager = None # Disable reset functionality
        
        # Initialize OpenCV window (if GUI is available)
        opencv_gui_available = False
        if not args.headless:
            try:
                # Check if OpenCV has GUI capabilities
                if hasattr(cv2, 'namedWindow') and hasattr(cv2, 'WINDOW_AUTOSIZE'):
                    cv2.namedWindow('VLA Test - Camera Views & Status', cv2.WINDOW_AUTOSIZE)
                    opencv_gui_available = True
                    print("✅ OpenCV visualization window created")
                else:
                    print("⚠️ OpenCV GUI functionality not available (missing GUI module)")
                    print("📊 Will use console output to display data")
                    opencv_gui_available = False
            except Exception as e:
                print(f"⚠️ OpenCV GUI not available: {e}")
                print("📊 Will use console output to display data")
                opencv_gui_available = False
        
        # Run evaluation rounds
        print(f"🎮 Starting {args.eval_rounds} evaluation round(s)...")
        
        # Create rate limiter
        rate_limiter = RateLimiter(args.step_hz)
        
        for round_idx in range(args.eval_rounds):
            print(f"\n📋 Round {round_idx + 1}/{args.eval_rounds}")

            # Calculate total steps
            total_steps = int(args.episode_length_s * args.step_hz)

            # Starting from the second round, reset the scene
            if round_idx > 0 and scene_manager is not None:
                print("\n🔄 Resetting scene...")
                # Reset object positions and robot pose
                scene_manager.reset_scene()
                # As per instructions, set all joint positions to 0
                robot.set_joint_positions(np.zeros(robot.num_dof))
                print("🤖 Robot joints have been reset to zero positions")
                
                # Wait for the scene to stabilize
                print("⏳ Waiting for scene to stabilize (60 frames)...")
                for _ in range(60):
                    world.step(render=not args.headless)
                print("✅ Scene is stable")
            
            # Reset action sequence index
            current_action_index = 0
            actions = None
            
            # [Important] Get initial observation before the loop starts
            print(" H Getting initial observation...")
            observation, front_img, wrist_img, joint_pos = get_observation_with_visualization(
                world, robot, camera_controller, args.policy_language_instruction
            )
            
            step = 0
            while step < total_steps:
                # Core logic refactored: fully synchronous blocking model
                # 1. At the start of each loop, if the action sequence is finished,
                #    block to get a new sequence.
                if actions is None or current_action_index >= args.policy_action_horizon:
                    print(f"🔄 Step {step+1}: Getting new action sequence (blocking)...")
                    start_time = time.time()
                    actions = vla_client.get_action(observation)
                    end_time = time.time()
                    print(f"✅ New action sequence received. Time taken: {end_time - start_time:.4f}s")
                    
                    if actions is not None and actions.shape[0] > 0:
                        current_action_index = 0
                        print(f"   Sequence shape: {actions.shape}")
                    else:
                        print(f"⚠️ Step {step+1}: No valid action sequence received. Skipping...")
                        # Even on failure, step and wait
                        world.step(render=not args.headless)
                        rate_limiter.sleep()
                        step += 1
                        continue
                
                # 2. Extract the current action from the sequence
                if current_action_index < actions.shape[0]:
                    if len(actions.shape) == 3:  # [50, 1, 6]
                        action = actions[current_action_index, :, :]
                    elif len(actions.shape) == 2:  # [50, 6]
                        action = actions[current_action_index, :].unsqueeze(0)
                    else:
                        print(f"⚠️ Unknown action tensor shape: {actions.shape}. Skipping...")
                        world.step(render=not args.headless)
                        rate_limiter.sleep()
                        step += 1
                        continue
                else:
                    print(f"⚠️ Action index out of bounds: {current_action_index}, Action shape: {actions.shape}. Skipping...")
                    world.step(render=not args.headless)
                    rate_limiter.sleep()
                    step += 1
                    continue

                # 3. Execute the action (set controller target)
                env_adapter.execute_action(action.unsqueeze(0))
                
                # 4. Step the physics world (let the action take effect)
                world.step(render=not args.headless)

                # 5. Get the new observation (result after action execution)
                observation, front_img, wrist_img, joint_pos = get_observation_with_visualization(
                    world, robot, camera_controller, args.policy_language_instruction
                )
                
                # 6. Record and visualize
                action_np = action.cpu().numpy().flatten()
                action_recorder.record_action(step+1, round_idx+1, joint_pos, action_np)
                
                if opencv_gui_available:
                    display_camera_images(front_img, wrist_img, step, joint_pos, action.unsqueeze(0), observation)
                else:
                    progress = f"{current_action_index + 1}/{args.policy_action_horizon}"
                    print(f"📊 Data for step {step+1} [Sequence Progress:{progress}]:")
                    print(f"   Joint Positions: {joint_pos}")
                    print(f"   Current Action: {action_np}")
                    if observation is not None:
                        print(f"   VLA Input Joint Pos: {observation['joint_pos'].cpu().numpy()}")
                        print(f"   Front Camera Shape: {observation['front'].shape}")
                        print(f"   Wrist Camera Shape: {observation['wrist'].shape}")

                # If configured to hide debug viz, continuously clear it
                if debug_clear_enabled and debug_visualizer is not None:
                    try:
                        if debug_visualizer.is_enabled:
                            debug_visualizer.is_enabled = False
                            debug_visualizer._toggle_ik_target_sphere()
                    except Exception:
                        pass

                # 7. Update counters
                current_action_index += 1
                step += 1
                
                # 8. Use RateLimiter for precise waiting
                rate_limiter.sleep()
            
            print(f"✅ Round {round_idx + 1} completed")
        
        # Cleanup
        print("\n🧹 Cleaning up resources...")
        vla_client.close()
        
        # Close action history recorder
        action_recorder.close()
        
        # Cleanup OpenCV windows
        if opencv_gui_available:
            cv2.destroyAllWindows()
            print("✅ OpenCV windows closed")
        
        # Import cleanup function
        from src.vla.environment_initializer import cleanup_environment
        
        # Cleanup environment (this will close the simulation_app)
        cleanup_environment(None)  # simulation_app is handled within cleanup_environment
        
        print("🎉 VLA inference test completed!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ VLA inference test failed: {e}")
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())
