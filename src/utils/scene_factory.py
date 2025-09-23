# -*- coding: utf-8 -*-
"""
Scene Factory Module
Provides functionality for constructing scenes, extracted from the main script.
"""

import os
import numpy as np
from .config_utils import ConfigManager


class SceneFactory:
    """Scene Factory Class"""
    
    def __init__(self, project_root, world):
        """Initializes the SceneFactory.
        
        Args:
            project_root (str): The root path of the project.
            world: The Isaac Sim World object.
        """
        self.project_root = project_root
        self.world = world
        self.config_manager = ConfigManager(project_root)
    
    def create_orange_plate_scene(self, scene_config):
        """Creates the orange and plate scene.
        
        Args:
            scene_config (dict): The scene configuration.
            
        Returns:
            dict: A dictionary of scene objects.
        """
        print("\n🍊 Creating the orange and plate scene...")
        scene_objects = {}
        
        # Get configuration parameters
        plate_config = self.config_manager.get_plate_config(scene_config)
        orange_config = self.config_manager.get_orange_config(scene_config)
        
        # Extract parameters
        plate_position = plate_config["position"]
        plate_radius = plate_config["radius"]
        plate_height = plate_config["height"]
        orange_count = orange_config["count"]
        orange_mass = orange_config["mass"]
        orange_usd_paths = orange_config["usd_paths"]
        x_range = orange_config["x_range"]
        y_range = orange_config["y_range"]
        z_drop_height = orange_config["z_drop_height"]
        orange_radius = orange_config["orange_radius"]
        min_distance = orange_config["min_distance"]
        max_attempts = orange_config["max_attempts"]
        
        print(f"✅ Configuration parameters loaded:")
        print(f"   Plate Position: {plate_position}")
        print(f"   Plate Radius: {plate_radius}m, Height: {plate_height}m")
        print(f"   Number of Oranges: {orange_count}, Mass: {orange_mass}kg")
        print(f"   Orange Generation Range: X{x_range}, Y{y_range}, Z={z_drop_height}")
        print(f"   Min Distance: {min_distance}m, Max Attempts: {max_attempts}")
        
        # Initialize the smart placement system
        from src.scene.smart_placement import SmartPlacement
        
        smart_placement = SmartPlacement(
            config_path="config/scene_config.yaml",
            plate_position=plate_position
        )
        
        # Set plate position
        print("🍽️ Setting plate position...")
        plate_center = plate_position.copy()
        print(f"✅ Using plate position from configuration file: {plate_center}")
        
        # Generate orange positions (avoiding the plate)
        print("🍊 Generating orange positions (avoiding the plate)...")
        smart_placement.clear_placement_history()
        plate_object_info = {
            "position": np.array(plate_center),
            "type": "plate", 
            "name": "plate_object"
        }
        smart_placement.placed_objects.append(plate_object_info)
        print(f"📍 Plate avoidance zone: Position {plate_center}, Radius {smart_placement.object_sizes['plate']['radius']}m")
        
        # Generate orange positions
        orange_types = ["orange"] * orange_count
        orange_names = [f"orange{i+1}_object" for i in range(orange_count)]
        orange_positions = smart_placement.generate_safe_positions(orange_types, orange_names)
        
        # Combine all positions
        safe_positions = orange_positions + [np.array(plate_center)]
        print(f"✅ Generated {len(orange_positions)} orange positions + 1 plate position.")
        
        # Load orange objects
        orange_objects_loaded = self._load_orange_objects(
            orange_count, orange_usd_paths, safe_positions, orange_mass
        )
        
        # Add oranges to the scene objects dictionary
        for name, obj in orange_objects_loaded.items():
            scene_objects[name] = obj
        
        # Load the plate object
        plate_obj = self._load_plate_object(plate_center, plate_radius, plate_height)
        if plate_obj:
            scene_objects["plate_object"] = plate_obj
        
        print(f"✅ Orange and plate scene created: {len(scene_objects)} objects.")
        for name in scene_objects.keys():
            print(f"    - {name}")
        
        return scene_objects, orange_positions, plate_center
    
    def _load_orange_objects(self, orange_count, orange_usd_paths, safe_positions, orange_mass):
        """Loads the orange objects.
        
        Args:
            orange_count (int): The number of oranges.
            orange_usd_paths (list): A list of paths to the orange USD files.
            safe_positions (list): A list of safe positions.
            orange_mass (float): The mass of the oranges.
            
        Returns:
            dict: A dictionary of orange objects.
        """
        orange_objects_loaded = {}
        
        for i in range(orange_count):
            if i < len(safe_positions) - 1:  # Subtract 1 because the last position is the plate
                usd_path = f"{self.project_root}/{orange_usd_paths[i]}" if i < len(orange_usd_paths) else f"{self.project_root}/{orange_usd_paths[0]}"
                prim_path = f"/World/orange{i+1}"
                scene_name = f"orange{i+1}_object"
                
                orange_obj = self._load_single_orange(usd_path, prim_path, safe_positions[i].tolist(), scene_name, orange_mass)
                if orange_obj:
                    orange_objects_loaded[scene_name] = orange_obj
                    print(f"✅ {scene_name} loaded successfully: Position {safe_positions[i].tolist()}, Mass {orange_mass}kg")
        
        return orange_objects_loaded
    
    def _load_single_orange(self, usd_path, prim_path, position, name, mass=0.15):
        """Loads a single orange.
        
        Args:
            usd_path (str): The path to the USD file.
            prim_path (str): The prim path.
            position (list): The position.
            name (str): The name of the object.
            mass (float): The mass.
            
        Returns:
            The orange object or None.
        """
        if not os.path.exists(usd_path):
            print(f"⚠️ Orange USD file not found: {usd_path}")
            return None
            
        try:
            print(f"🔧 Loading orange USD: {os.path.basename(usd_path)}")
            
            # Step 1: Load the USD to the stage
            from isaacsim.core.utils.stage import add_reference_to_stage
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            print(f"✅ Orange USD loaded to stage: {prim_path}")
            
            # Step 2: Create a physics object using SingleRigidPrim
            from isaacsim.core.prims import SingleRigidPrim
            
            orange = self.world.scene.add(
                SingleRigidPrim(
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    mass=mass
                )
            )
            
            print(f"✅ Orange loaded: {name} at position {position} with mass {mass}kg")
            return orange
            
        except Exception as e:
            print(f"❌ Failed to load orange {name}: {e}")
            return None
    
    def _load_plate_object(self, plate_center, plate_radius, plate_height):
        """Loads the plate object.
        
        Args:
            plate_center (list): The center position of the plate.
            plate_radius (float): The radius of the plate.
            plate_height (float): The height of the plate.
            
        Returns:
            The plate object or a virtual plate object.
        """
        print("🍽️ Loading plate USD model...")
        print(f"📍 Using plate position: {plate_center}")
        
        plate_usd_path = f"{self.project_root}/assets/objects/Plate/Plate.usd"
        
        try:
            print(f"🔧 Loading plate USD: {os.path.basename(plate_usd_path)}")
            from isaacsim.core.utils.stage import add_reference_to_stage
            add_reference_to_stage(usd_path=plate_usd_path, prim_path="/World/plate_object")
            print("✅ Plate USD loaded to stage: /World/plate_object")
            
            # Create a physics object for the plate using SingleRigidPrim
            from isaacsim.core.prims import SingleRigidPrim
            plate = self.world.scene.add(
                SingleRigidPrim(
                    prim_path="/World/plate_object",
                    name="plate_object",
                    position=plate_center,
                    mass=0.2  # 200g mass for the plate
                )
            )
            print(f"✅ Plate loaded: plate_object at position {plate_center} with mass 0.2kg")
            return plate
            
        except Exception as e:
            print(f"❌ Failed to load plate: {e}")
            print("🔄 Using a virtual plate object as a fallback...")
            
            # Create a virtual plate object
            return self._create_virtual_plate(plate_center, plate_radius, plate_height)
    
    def _create_virtual_plate(self, position, radius=0.1, height=0.02):
        """Creates a virtual plate object.
        
        Args:
            position (list): The position.
            radius (float): The radius.
            height (float): The height.
            
        Returns:
            A virtual plate object.
        """
        class VirtualPlateObject:
            def __init__(self, position, radius=0.1, height=0.02):
                self.position = np.array(position)
                self.radius = radius
                self.height = height
            
            def get_world_pose(self):
                return self.position, np.array([1, 0, 0, 0])  # Position and identity quaternion
            
            def set_world_pose(self, position, orientation=None):
                self.position = np.array(position)
                print(f"🍽️ Virtual plate position updated: [{self.position[0]:.4f}, {self.position[1]:.4f}, {self.position[2]:.4f}]")
            
            def get_linear_velocity(self):
                return np.array([0, 0, 0])  # Stationary state
        
        virtual_plate = VirtualPlateObject(position, radius=radius, height=height)
        print(f"✅ Virtual plate object created at position: {position}")
        return virtual_plate


# Compatibility function to maintain the same interface as the main script.
def create_orange_plate_scene(project_root, world, scene_config):
    """Compatibility function."""
    factory = SceneFactory(project_root, world)
    return factory.create_orange_plate_scene(scene_config)
