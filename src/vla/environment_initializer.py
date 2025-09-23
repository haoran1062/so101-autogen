# -*- coding: utf-8 -*-
"""
Environment Initializer
Extracts the environment initialization logic from the main script for VLA testing.
"""

import os
import sys
import time
import logging

logger = logging.getLogger(__name__)


def initialize_environment_for_vla(project_root: str = None, headless: bool = False):
    """
    Initializes the environment for VLA testing.
    This function extracts the environment initialization logic from the main script.
    
    Args:
        project_root: The root path of the project.
        headless: Whether to use headless mode.
        
    Returns:
        tuple: (world, robot, camera_controller, config)
    """
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    print("🚀 Initializing environment for VLA testing")
    print("=" * 50)
    
    try:
        # 1. Load configuration
        print("📋 Step 1: Loading configuration")
        from src.config.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        print("✅ Configuration loaded successfully.")
        
        # 2. Start Isaac Sim simulation
        print("\n🎮 Step 2: Starting Isaac Sim simulation")
        from src.core.simulation_manager import SimulationManager
        sim_manager = SimulationManager(headless=headless)  # Use headless mode based on the argument
        simulation_app = sim_manager.start_simulation()
        print(f"✅ Isaac Sim simulation started (Headless: {headless})")
        
        # 3. Load Isaac Sim extensions and modules
        print("\n🔧 Step 3: Loading Isaac Sim extensions and modules")
        from src.utils.extension_loader import ExtensionLoader
        extension_modules = ExtensionLoader.load_all()
        print("✅ Isaac Sim extensions and modules loaded successfully.")
        
        # 4. Import Isaac Sim related modules (after extension loading)
        from src.core.world_setup import WorldSetup
        from src.scene.scene_manager import SceneManager
        
        # 5. Create the World and scene
        print("\n🌍 Step 4: Creating the World and scene")
        world_setup = WorldSetup(config)
        world = world_setup.create_world()
        world_setup.setup_environment()
        world_setup.add_follow_target_task()
        print("✅ World and scene created successfully.")
        
        # 6. Load scene objects (oranges and plate)
        print("\n🍊 Step 5: Loading scene objects")
        from src.utils.scene_factory import SceneFactory
        from src.utils.config_utils import load_scene_config
        
        scene_config = load_scene_config(project_root)
        scene_factory = SceneFactory(project_root, world)
        scene_objects, orange_positions, plate_center = scene_factory.create_orange_plate_scene(scene_config)
        print("✅ Scene objects loaded successfully.")
        
        # 7. Reset the world and initialize the task
        print("\n🔄 Step 6: Resetting the world and initializing the task")
        world.reset()
        
        # Wait for the task to initialize
        print("⏳ Waiting for task initialization...")
        for i in range(60):
            world.step(render=not headless)
            if i % 20 == 0:
                print(f"   Initialization progress: {i+1}/60 steps")
        
        # 8. Get the robot object
        print("\n🤖 Step 7: Getting the robot object")
        task = world.get_task("so101_follow_target")
        task_params = task.get_params()
        robot_name = task_params["robot_name"]["value"]
        robot = world.scene.get_object(robot_name)
        
        if robot is None:
            raise RuntimeError(f"Robot object not found: {robot_name}")
        print(f"✅ Robot object '{robot.name}' acquired successfully.")
        
        # 9. Create the camera controller
        print("\n📷 Step 8: Creating the camera controller")
        try:
            from src.camera import get_multi_camera_controller_from_ref
            MultiCameraController = get_multi_camera_controller_from_ref()
            camera_controller = MultiCameraController(config=scene_config)
            print("✅ Camera controller created (parameters loaded from config).")
            
            # Wait for camera data to be ready
            print("⏳ Waiting for camera data to be ready...")
            for i in range(30):  # Wait for 30 steps
                world.step(render=not headless)
                if i % 10 == 0:
                    print(f"   Camera initialization progress: {i+1}/30 steps")
            
            # Verify that camera data is available
            if camera_controller and hasattr(camera_controller, 'front_camera') and camera_controller.front_camera:
                try:
                    test_image = camera_controller.front_camera.get_rgba()
                    if test_image is not None and len(test_image.shape) == 3:
                        print("✅ Front camera data is ready.")
                    else:
                        print(f"⚠️ Front camera data format is unexpected: {test_image.shape if test_image is not None else 'None'}")
                except Exception as e:
                    print(f"⚠️ Failed to get front camera data: {e}")
            
            if camera_controller and hasattr(camera_controller, 'wrist_camera') and camera_controller.wrist_camera:
                try:
                    test_image = camera_controller.wrist_camera.get_rgba()
                    if test_image is not None and len(test_image.shape) == 3:
                        print("✅ Wrist camera data is ready.")
                    else:
                        print(f"⚠️ Wrist camera data format is unexpected: {test_image.shape if test_image is not None else 'None'}")
                except Exception as e:
                    print(f"⚠️ Failed to get wrist camera data: {e}")
                    
        except Exception as e:
            print(f"⚠️ Failed to create camera controller: {e}")
            camera_controller = None
        
        print("\n✅ Environment initialization complete.")
        print("=" * 50)
        
        return world, robot, camera_controller, config
        
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def cleanup_environment(simulation_app=None):
    """
    Cleans up environment resources.
    
    Args:
        simulation_app: The simulation application object.
    """
    print("\n🧹 Cleaning up environment resources...")
    try:
        if simulation_app:
            simulation_app.close()
        print("✅ Environment resources cleaned up successfully.")
    except Exception as e:
        print(f"⚠️ An error occurred while cleaning up environment resources: {e}")
