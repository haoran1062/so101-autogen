# -*- coding: utf-8 -*-
"""
Scene Manager
Uniformly manages scene resets, object regeneration, and integrates the
smart placement system and random position generator.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any

# Relative imports
from .smart_placement import SmartPlacement
from .random_generator import RandomPositionGenerator

logger = logging.getLogger(__name__)

class SceneManager:
    """Manages scene resets and object placements."""
    
    def __init__(self, config: Dict[str, Any], world=None):
        """
        Initializes the SceneManager.
        
        Args:
            config: The configuration dictionary.
            world: The Isaac Sim World object.
        """
        self.config = config
        self.world = world
        
        # Read the plate position from the configuration file
        self.plate_position = config.get('scene', {}).get('plate', {}).get('position', [0.28, 0.0, 0.1])
        
        # Initialize the smart placement system
        self.smart_placement = SmartPlacement(plate_position=self.plate_position)
        
        # Initialize the random generator (passing orange generation config and plate position)
        orange_config = config.get('scene', {}).get('oranges', {}).get('generation', {})
        self.random_generator = RandomPositionGenerator(orange_config, self.plate_position)
        
        # Records for scene objects
        self.scene_objects = {}  # Format: {"orange1": object, "plate": object, ...}
        self.object_initial_positions = {}  # Record of initial positions
        
        # Read the plate position from the configuration file
        self.plate_position = None
        if config and "scene" in config and "plate" in config["scene"]:
            self.plate_position = config["scene"]["plate"]["position"].copy()
            logger.info(f"✅ Plate position loaded from config: {self.plate_position}")
        
        logger.info("✅ SceneManager initialized.")
    
    def set_world(self, world):
        """Sets the World object."""
        self.world = world
        logger.info("✅ World object has been set in SceneManager.")
    
    def register_scene_objects(self, objects: Dict[str, Any]):
        """
        Registers objects present in the scene.
        
        Args:
            objects: A dictionary of objects, e.g., {"orange1": object, "plate": object}.
        """
        self.scene_objects.update(objects)
        
        # Record initial positions
        for name, obj in objects.items():
            if obj and hasattr(obj, 'get_world_pose'):
                try:
                    position, _ = obj.get_world_pose()
                    self.object_initial_positions[name] = position
                    logger.debug(f"📍 Recorded initial position for {name}: {position}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get initial position for {name}: {e}")
        
        logger.info(f"✅ Registered {len(objects)} scene objects.")
    
    def set_orange_reset_positions(self, positions: Dict[str, List[float]]):
        """
        Sets the reset positions for the oranges.
        
        Args:
            positions: A dictionary of positions, e.g., {"orange1_object": [x, y, z], ...}.
        """
        self.object_initial_positions.update(positions)
        logger.info(f"✅ Set reset positions for {len(positions)} oranges.")
        for name, pos in positions.items():
            logger.info(f"    - {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    def reset_scene(self):
        """
        Resets the scene (e.g., on 'R' key press).
        1. Generates new random positions.
        2. Uses smart placement to avoid overlaps.
        3. Updates the positions of the objects.
        """
        logger.info("🔄 Starting scene reset...")
        print("🔄 Resetting scene...")
        
        try:
            # 1. Clear smart placement history
            self.smart_placement.clear_placement_history()
            
            # 2. Separate oranges and plates
            orange_names = []
            plate_names = []
            
            for name in self.scene_objects.keys():
                if "orange" in name.lower():
                    orange_names.append(name)
                elif "plate" in name.lower():
                    plate_names.append(name)
            
            if not orange_names and not plate_names:
                logger.warning("⚠️ No objects found to reset.")
                return False
            
            reset_count = 0
            
            # 3. Handle plate position first (prioritizing config file)
            plate_center = None
            if plate_names:
                # Prioritize using the plate position from the config file
                if hasattr(self, 'plate_position') and self.plate_position is not None:
                    plate_center = self.plate_position.copy()
                    logger.info(f"✅ Using plate position from config: {plate_center}")
                else:
                    # Fallback to smart placement
                    plate_types = ["plate"] * len(plate_names)
                    plate_positions = self.smart_placement.generate_safe_positions(plate_types, plate_names)
                    
                    if len(plate_positions) >= 1:
                        plate_center = plate_positions[0].tolist()
                        logger.info(f"✅ Plate smart placement successful: {plate_center}")
                    else:
                        # Fallback to default position from config
                        default_plate_pos = self.config.get('scene', {}).get('plate', {}).get('position', [0.28, 0.0, 0.1])
                        plate_center = default_plate_pos.copy()
                        logger.warning(f"⚠️ Plate smart placement failed, using default config position: {plate_center}")
                
                # Add the plate to the avoidance list for other objects
                self.smart_placement.clear_placement_history()
                plate_object_info = {
                    "position": np.array(plate_center),
                    "type": "plate", 
                    "name": plate_names[0]
                }
                self.smart_placement.placed_objects.append(plate_object_info)
                
                # Update the plate's position
                if self._update_object_position(plate_names[0], np.array(plate_center)):
                    reset_count += 1
            
            # 4. Handle orange positions (avoiding the plate)
            orange_positions = []
            if orange_names:
                orange_types = ["orange"] * len(orange_names)
                orange_positions = self.smart_placement.generate_safe_positions(orange_types, orange_names)
                
                # Update orange positions
                for i, (name, new_pos) in enumerate(zip(orange_names[:len(orange_positions)], orange_positions)):
                    if self._update_object_position(name, new_pos):
                        reset_count += 1
            
            total_objects = len(orange_names) + len(plate_names)
            successful_placements = len(orange_positions) + (1 if plate_center else 0)
            if successful_placements != total_objects:
                logger.warning(f"⚠️ Only {successful_placements}/{total_objects} safe positions were successfully generated.")
            
            # 5. Print detailed debug information
            all_object_names = orange_names + plate_names
            self._print_debug_info(plate_center, orange_positions, all_object_names)
            
            logger.info(f"✅ Scene reset complete: {reset_count}/{total_objects} objects have been reset.")
            print(f"✅ Scene reset complete: {reset_count}/{total_objects} objects have been reset.")
            
            return reset_count > 0
            
        except Exception as e:
            logger.error(f"❌ Scene reset failed: {e}")
            print(f"❌ Scene reset failed: {e}")
            return False
    
    def _update_object_position(self, object_name: str, new_position: np.ndarray) -> bool:
        """
        Updates the position of a single object.
        
        Args:
            object_name: The name of the object.
            new_position: The new position.
            
        Returns:
            True if successful, False otherwise.
        """
        if object_name not in self.scene_objects:
            logger.warning(f"⚠️ Object {object_name} is not registered.")
            return False
        
        obj = self.scene_objects[object_name]
        if obj is None:
            logger.warning(f"⚠️ Object {object_name} is null.")
            return False
        
        try:
            # Set new position and orientation
            obj.set_world_pose(
                position=new_position,
                orientation=np.array([1.0, 0.0, 0.0, 0.0])  # Default orientation
            )
            
            # Reset velocities to ensure stability
            if hasattr(obj, 'set_linear_velocity'):
                obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            if hasattr(obj, 'set_angular_velocity'):
                obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
            
            logger.debug(f"✅ Position of {object_name} updated to: [{new_position[0]:.3f}, {new_position[1]:.3f}, {new_position[2]:.3f}]")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update position for {object_name}: {e}")
            return False
    
    def reset_to_initial_positions(self):
        """Resets all objects to their initial positions."""
        logger.info("🔄 Resetting objects to their initial positions...")
        
        reset_count = 0
        for name, initial_pos in self.object_initial_positions.items():
            if self._update_object_position(name, initial_pos):
                reset_count += 1
        
        logger.info(f"✅ {reset_count}/{len(self.object_initial_positions)} objects have been reset to their initial positions.")
        print(f"✅ {reset_count}/{len(self.object_initial_positions)} objects have been reset to their initial positions.")
        
        return reset_count > 0
    
    def generate_random_positions_only(self, num_oranges: int = 3) -> List[np.ndarray]:
        """
        Generates random positions only (without updating objects).
        Uses the existing random generator.
        
        Args:
            num_oranges: The number of oranges.
            
        Returns:
            A list of positions.
        """
        try:
            # Use the existing random generator
            positions_list = self.random_generator.generate_random_orange_positions(num_oranges)
            # Convert to numpy array format
            positions = [np.array(pos) for pos in positions_list]
            logger.info(f"🎲 Generated {len(positions)} random positions.")
            return positions
        except Exception as e:
            logger.error(f"❌ Failed to generate random positions: {e}")
            return []
    
    def get_scene_info(self) -> Dict:
        """Gets information about the scene."""
        return {
            "registered_objects": list(self.scene_objects.keys()),
            "object_count": len(self.scene_objects),
            "placement_info": self.smart_placement.get_placement_info(),
            "has_world": self.world is not None
        }
    
    def get_oranges(self) -> List[Any]:
        """Gets all registered orange objects."""
        orange_objects = []
        for name, obj in self.scene_objects.items():
            if "orange" in name.lower():
                orange_objects.append(obj)
        return orange_objects
    
    def _print_debug_info(self, plate_center, orange_positions, object_names):
        """Prints detailed debug information."""
        print("\n" + "="*60)
        print("🔍 Reset Debug Information")
        print("="*60)
        
        # 1. Plate area information
        if plate_center:
            plate_radius = 0.10  # 10cm radius
            plate_x, plate_y = plate_center[0], plate_center[1]
            
            print(f"🍽️ Plate Area:")
            print(f"   Center Position: [{plate_x:.3f}, {plate_y:.3f}, {plate_center[2]:.3f}]")
            print(f"   Radius: {plate_radius:.2f}m (10cm)")
            print(f"   X Range: [{plate_x-plate_radius:.3f}, {plate_x+plate_radius:.3f}]")
            print(f"   Y Range: [{plate_y-plate_radius:.3f}, {plate_y+plate_radius:.3f}]")
        
        # 2. Orange position information
        print(f"\n🍊 Orange Positions:")
        orange_count = 0
        for i, name in enumerate(object_names):
            if "orange" in name.lower() and orange_count < len(orange_positions):
                pos = orange_positions[orange_count]
                print(f"   {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                orange_count += 1
        
        # 3. Overlap detection
        overlap_detected = False
        if plate_center and orange_positions:
            print(f"\n⚠️ Overlap Detection:")
            plate_radius = 0.10
            orange_count = 0
            
            for i, name in enumerate(object_names):
                if "orange" in name.lower() and orange_count < len(orange_positions):
                    pos = orange_positions[orange_count]
                    
                    # Calculate distance in the XY plane
                    distance_xy = np.sqrt((pos[0] - plate_center[0])**2 + (pos[1] - plate_center[1])**2)
                    
                    # Check if it's within the plate's XY area
                    is_in_plate_xy = distance_xy <= plate_radius
                    if is_in_plate_xy:
                        overlap_detected = True
                    
                    print(f"   {name}:")
                    print(f"     Distance to plate center (XY): {distance_xy:.3f}m")
                    print(f"     Within plate's XY area: {'❌ Yes' if is_in_plate_xy else '✅ No'}")
                    
                    orange_count += 1
        
        print("="*60 + "\n")
    
    def _check_orange_plate_overlap(self, plate_center, orange_positions):
        """Checks if any orange overlaps with the plate."""
        if not plate_center or not orange_positions:
            return False
        
        plate_radius = 0.10  # 10cm radius
        
        for pos in orange_positions:
            # Calculate distance in the XY plane
            distance_xy = np.sqrt((pos[0] - plate_center[0])**2 + (pos[1] - plate_center[1])**2)
            
            # Check if it's within the plate's XY area
            if distance_xy <= plate_radius:
                return True
        
        return False
