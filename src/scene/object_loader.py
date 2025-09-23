# -*- coding: utf-8 -*-
"""
Object Loader - Manages the loading of objects into the scene.

"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
import logging

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleRigidPrim

import omni.usd
from pxr import Sdf, Gf, UsdGeom

from .random_generator import RandomPositionGenerator

# Virtual plate object class for placement detection when the actual USD is unavailable.
class VirtualPlateObject:
    """A virtual plate object for placement detection when the actual USD is not available."""
    def __init__(self, position, radius=0.1, height=0.02):
        self.position = np.array(position)
        self.radius = radius
        self.height = height
        self.name = "virtual_plate_object"
        print(f"🍽️ Virtual plate object initialized: Position {self.position.tolist()}, Radius {self.radius}m, Height {self.height}m")

    def get_world_pose(self):
        """Returns the position and an identity quaternion."""
        return self.position, np.array([1.0, 0.0, 0.0, 0.0])  # Position and identity quaternion

    def set_world_pose(self, position, orientation=np.array([1.0, 0.0, 0.0, 0.0])):
        """Sets the world pose - supports position reset."""
        self.position = np.array(position)
        print(f"🍽️ Virtual plate position updated: [{self.position[0]:.4f}, {self.position[1]:.4f}, {self.position[2]:.4f}]")

    def get_linear_velocity(self):
        """Returns a zero velocity vector, indicating it is stationary."""
        return np.array([0.0, 0.0, 0.0])

class ObjectLoader:
    """
    Object Loader
    Loads oranges and the plate into the scene, following the methodology of the original main script.
    """
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        """
        Initializes the ObjectLoader.
        
        Args:
            config: Scene configuration.
            project_root: The root path of the project.
        """
        self.config = config
        self.project_root = project_root
        
        # Orange configuration
        oranges_config = config.get('scene', {}).get('oranges', {})
        self.orange_models = oranges_config.get('models', ["Orange001", "Orange002", "Orange003"])
        self.orange_usd_paths = oranges_config.get('usd_paths', [
            "assets/objects/Orange001/Orange001.usd",
            "assets/objects/Orange002/Orange002.usd", 
            "assets/objects/Orange003/Orange003.usd"
        ])
        self.orange_count = oranges_config.get('count', 3)
        self.orange_mass = oranges_config.get('physics', {}).get('mass', 0.15)
        
        # Plate configuration
        plate_config = config.get('scene', {}).get('plate', {})
        self.plate_usd_path = plate_config.get('usd_path', 'assets/objects/Plate/Plate.usd')
        self.plate_position = plate_config.get('position', [0.25, -0.15, 0.1])
        self.plate_scale = plate_config.get('scale', 1.0)
        self.use_virtual_plate = plate_config.get('use_virtual', True)
        self.virtual_plate_config = plate_config.get('virtual_config', {})
        
        # Random position generator
        generation_config = oranges_config.get('generation', {})
        self.position_generator = RandomPositionGenerator(generation_config)
        
        # Store loaded objects
        self.orange_objects = {}
        self.orange_reset_positions = {}
        self.plate_object = None
        
        print(f"✅ Object Loader initialized.")
        print(f"   Number of oranges: {self.orange_count}")
        print(f"   Orange models: {self.orange_models}")
        print(f"   Plate position: {self.plate_position}")
        print(f"   Using virtual plate: {self.use_virtual_plate}")
    
    def load_orange(self, world: World, usd_path: str, prim_path: str, position: List[float], name: str, mass: float = 0.15):
        """
        Loads a single orange into the scene.
        """
        full_usd_path = os.path.join(self.project_root, usd_path)
        
        if not os.path.exists(full_usd_path):
            print(f"❌ Orange USD file not found: {full_usd_path}")
            return None
            
        try:
            print(f"🔧 Loading orange USD: {full_usd_path}")
            
            # Step 1: Load the USD to the stage
            add_reference_to_stage(usd_path=full_usd_path, prim_path=prim_path)
            print(f"✅ Orange USD loaded to stage: {prim_path}")
            
            # Step 2: Add as a SingleRigidPrim
            orange = world.scene.add(
                SingleRigidPrim(
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    mass=mass  # 150g mass for the orange
                )
            )
            
            print(f"✅ Orange loaded successfully: {name} at position {position} with mass {mass}kg")
            return orange
            
        except Exception as e:
            print(f"❌ Failed to load orange {name}: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def load_oranges(self, world: World) -> Dict[str, Any]:
        """
        Loads all oranges into the scene.
        """
        try:
            print(f"🍊 Loading {self.orange_count} oranges for the grasp test...")
            
            # Generate random positions
            print(f"🎲 Generating {self.orange_count} random positions for oranges...")
            random_positions = self.position_generator.generate_random_orange_positions(self.orange_count)
            
            # Ensure at least 3 positions are set for compatibility
            orange1_reset_pos = random_positions[0] if len(random_positions) > 0 else [0.2, 0.1, 0.1]
            orange2_reset_pos = random_positions[1] if len(random_positions) > 1 else [0.25, 0.15, 0.1] 
            orange3_reset_pos = random_positions[2] if len(random_positions) > 2 else [0.15, 0.05, 0.1]
            
            positions = [orange1_reset_pos, orange2_reset_pos, orange3_reset_pos]
            
            for i, pos in enumerate(positions[:3]):
                print(f"🍊 Orange {i+1} random position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            # Load the oranges
            orange_objects = {}
            orange_reset_positions = {}
            
            for i in range(min(self.orange_count, len(self.orange_usd_paths))):
                usd_path = self.orange_usd_paths[i]
                model_name = self.orange_models[i]
                prim_path = f"/World/orange{i+1}"
                object_name = f"orange{i+1}_object"
                position = positions[i]
                
                # Load using the helper function
                orange = self.load_orange(world, usd_path, prim_path, position, object_name, self.orange_mass)
                
                if orange is not None:
                    orange_objects[object_name] = orange
                    orange_reset_positions[object_name] = position
                    print(f"✅ Orange loaded: {object_name}")
            
            self.orange_objects = orange_objects
            self.orange_reset_positions = orange_reset_positions
            
            return {
                'objects': orange_objects,
                'reset_positions': orange_reset_positions
            }
            
        except Exception as e:
            print(f"❌ Failed to load oranges: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return {'objects': {}, 'reset_positions': {}}
    
    def load_plate(self, world: World) -> Optional[Any]:
        """
        Loads the plate into the scene.
        """
        try:
            print("📦 Loading the plate...")
            
            # Create a virtual plate object for placement detection.
            print("🔧 Creating a virtual plate object for placement detection...")
            
            # Use plate position from the configuration.
            virtual_config = self.virtual_plate_config
            plate_center = virtual_config.get('position', [0.25, -0.15, 0.005])
            plate_radius = virtual_config.get('radius', 0.1)
            plate_height = virtual_config.get('height', 0.02)
            
            plate_object = VirtualPlateObject(plate_center, plate_radius, plate_height)
            
            print(f"✅ Virtual plate object created: Position {plate_center}, Radius {plate_radius}m, Height {plate_height}m")
            
            # If the actual plate USD file exists, attempt to load it.
            full_plate_usd_path = os.path.join(self.project_root, self.plate_usd_path)
            
            if os.path.exists(full_plate_usd_path):
                try:
                    print(f"🔧 Attempting to load actual plate USD: {full_plate_usd_path}")
                    
                    # Step 1: Load the USD to the stage.
                    add_reference_to_stage(usd_path=full_plate_usd_path, prim_path="/World/plate")
                    print("✅ Plate USD loaded to stage.")
                    
                    # Step 2: Set the scale.
                    stage = omni.usd.get_context().get_stage()
                    plate_prim = stage.GetPrimAtPath("/World/plate")
                    if plate_prim.IsValid():
                        xformable = UsdGeom.Xformable(plate_prim)
                        existing_ops = xformable.GetOrderedXformOps()
                        scale_op = None
                        for op in existing_ops:
                            if op.GetOpName() == "xformOp:scale":
                                scale_op = op
                                break
                        
                        if scale_op is None:
                            scale_op = xformable.AddScaleOp()
                        
                        scale_op.Set(Gf.Vec3f(self.plate_scale, self.plate_scale, self.plate_scale))
                        print(f"✅ Plate scale set to: {self.plate_scale}")
                    
                    # Step 3: Add as a SingleRigidPrim.
                    actual_plate = world.scene.add(
                        SingleRigidPrim(
                            prim_path="/World/plate",
                            name="plate_object",
                            position=plate_center,  # Use the corrected position
                            mass=0.5
                        )
                    )
                    
                    # If the actual plate is loaded successfully, use it.
                    plate_object = actual_plate
                    
                    print(f"✅ Actual plate loaded: Position {plate_center}, Scale {self.plate_scale}")
                    
                except Exception as e:
                    print(f"❌ Failed to load actual plate: {e}")
                    import traceback
                    print(f"Detailed error: {traceback.format_exc()}")
                    print("🔄 Continuing with the virtual plate object.")
            else:
                print(f"⚠️ Plate USD file not found: {full_plate_usd_path}")
                print("🔄 Using virtual plate object for placement detection.")
            
            self.plate_object = plate_object
            return plate_object
            
        except Exception as e:
            print(f"❌ Failed to load plate: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def regenerate_orange_positions(self, world: World):
        """
        Regenerates random positions for the oranges.
        """
        if not self.orange_objects:
            print("⚠️ No orange objects to reposition.")
            return
        
        print("🔄 Regenerating orange positions...")
        
        # Generate new orange positions.
        print("🎲 Generating new positions for the oranges.")
        new_positions = self.position_generator.generate_random_orange_positions(len(self.orange_objects))
        
        # Move oranges to their new positions.
        repositioned_count = 0
        for i, (name, orange_obj) in enumerate(self.orange_objects.items()):
            if i < len(new_positions) and orange_obj is not None:
                try:
                    new_pos = new_positions[i]
                    orange_obj.set_world_pose(
                        position=np.array(new_pos),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0])
                    )
                    orange_obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                    orange_obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                    
                    # Update the reset position.
                    self.orange_reset_positions[name] = new_pos
                    
                    print(f"✅ {name} moved to new random position: [{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}]")
                    repositioned_count += 1
                except Exception as e:
                    print(f"❌ Failed to update position for {name}: {e}")
        
        print(f"🎲 Random repositioning complete. Successfully moved {repositioned_count} oranges.")
    
    def get_orange_objects(self) -> Dict[str, Any]:
        """Gets the dictionary of orange objects."""
        return self.orange_objects
    
    def get_orange_reset_positions(self) -> Dict[str, List[float]]:
        """Gets the dictionary of orange reset positions."""
        return self.orange_reset_positions
    
    def get_plate_object(self) -> Any:
        """Gets the plate object."""
        return self.plate_object