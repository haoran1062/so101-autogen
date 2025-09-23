# -*- coding: utf-8 -*-
"""
Smart Object Placement System
A rewritten version to prevent object overlaps and ensure plausible placements.
All parameters are read from the configuration file to eliminate hardcoding.
"""

import numpy as np
import logging
import yaml
import os
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

def load_placement_config(config_path: str = "config/scene_config.yaml") -> Dict[str, Any]:
    """Loads the placement system configuration."""
    try:
        if not os.path.exists(config_path):
            logger.warning(f"⚠️ Configuration file not found: {config_path}. Using default settings.")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('placement', {})
    except Exception as e:
        logger.error(f"❌ Failed to load placement configuration: {e}. Using default settings.")
        return {}

def get_config_value(config: Dict, path: str, default: Any) -> Any:
    """Safely retrieves a value from a nested configuration dictionary."""
    keys = path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

class SmartPlacement:
    """Smart object placement manager."""
    
    def __init__(self, config_path: str = "config/scene_config.yaml", plate_position: List[float] = None):
        """
        Initializes the smart placement system.
        
        Args:
            config_path: Path to the configuration file.
            plate_position: The position of the plate, used for dynamic avoidance calculation.
        """
        # Load configuration
        self.config = load_placement_config(config_path)
        
        # Read workspace boundaries from config
        workspace_config = self.config.get('workspace_bounds', {})
        self.workspace_bounds = {
            "x": tuple(workspace_config.get('x', [0.05, 0.45])),   
            "y": tuple(workspace_config.get('y', [-0.30, 0.30])),  
            "z": tuple(workspace_config.get('z', [0.05, 0.10]))    
        }
        
        # Read robot exclusion zone from config
        robot_config = self.config.get('robot_exclusion_zone', {})
        self.robot_exclusion_zone = {
            "x": tuple(robot_config.get('x', [-0.1, 0.1])),    
            "y": tuple(robot_config.get('y', [-0.05, 0.05])),  
            "z": tuple(robot_config.get('z', [0.0, 0.5]))      
        }
        
        # Read object size configurations from config
        sizes_config = self.config.get('object_sizes', {})
        self.object_sizes = {}
        for obj_type in ['orange', 'plate', 'default']:
            size_cfg = sizes_config.get(obj_type, {})
            self.object_sizes[obj_type] = {
                'radius': size_cfg.get('radius', 0.05),
                'height': size_cfg.get('height', 0.10)
            }
        
        # Read safety distances from config
        safety_config = self.config.get('safety_distances', {})
        self.min_distance_between_objects = safety_config.get('min_distance_between_objects', 0.06)
        self.min_distance_from_edge = safety_config.get('min_distance_from_edge', 0.05)
        self.min_distance_from_robot = safety_config.get('min_distance_from_robot', 0.12)
        self.min_distance_from_plate = safety_config.get('min_distance_from_plate', 0.12)  # New: min distance from the plate
        
        # Read robot base position from config
        robot_base_config = self.config.get('robot_base_position', [0.0, 0.0, 0.0])
        self.robot_base_position = np.array(robot_base_config)
        
        # Dynamic plate exclusion zone
        self.plate_position = plate_position
        
        # Record of placed objects
        self.placed_objects = []  # Format: [{"position": [x,y,z], "type": "orange", "name": "orange1"}, ...]
        
        logger.info("✅ Smart placement system initialized.")
        logger.info(f"    - Workspace: X{self.workspace_bounds['x']}, Y{self.workspace_bounds['y']}, Z{self.workspace_bounds['z']}")
        logger.info(f"    - Robot Exclusion Zone: X{self.robot_exclusion_zone['x']}, Y{self.robot_exclusion_zone['y']}")
        if self.plate_position:
            logger.info(f"    - Plate Position: {self.plate_position}")
        logger.info(f"    - Min distance between objects: {self.min_distance_between_objects*100:.1f}cm")
    
    def set_plate_position(self, plate_position: List[float]):
        """Sets the plate position for dynamic avoidance calculation."""
        self.plate_position = plate_position
        logger.info(f"📍 Plate position has been updated: {self.plate_position}")
    
    def generate_safe_positions(self, object_types: List[str], 
                               object_names: List[str] = None,
                               max_attempts: int = 100) -> List[np.ndarray]:
        """
        Generates safe, non-overlapping positions for multiple objects.
        
        Args:
            object_types: A list of object types, e.g., ["orange", "orange", "plate"].
            object_names: An optional list of object names.
            max_attempts: The maximum number of attempts for each object.
            
        Returns:
            A list of positions. Returns an empty list if generation fails.
        """
        if object_names is None:
            object_names = [f"{obj_type}_{i}" for i, obj_type in enumerate(object_types)]
        
        if len(object_types) != len(object_names):
            logger.error("❌ The lengths of object_types and object_names lists do not match.")
            return []
        
        # Maximum number of overall attempts
        max_overall_attempts = 50
        overall_attempt = 0
        
        while overall_attempt < max_overall_attempts:
            overall_attempt += 1
            positions = []
            placed_objects = []
            all_success = True
            
            # Attempt to find a position for all objects
            for obj_type, obj_name in zip(object_types, object_names):
                position = self._find_safe_position(obj_type, placed_objects, max_attempts)
                
                if position is not None:
                    positions.append(position)
                    placed_objects.append({
                        "position": position,
                        "type": obj_type,
                        "name": obj_name
                    })
                    logger.debug(f"✅ Position for {obj_name}({obj_type}): [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
                else:
                    logger.debug(f"❌ Overall attempt #{overall_attempt}: Could not find a safe position for {obj_name}({obj_type}).")
                    all_success = False
                    break
            
            # If all objects were placed successfully, perform a final verification
            if all_success and len(positions) == len(object_types):
                # Verify that all positions are indeed non-overlapping (including with existing objects)
                if self._verify_all_positions_safe(positions, object_types):
                    # Update internal records
                    self.placed_objects.extend(placed_objects)
                    logger.info(f"🎯 Smart placement complete: After {overall_attempt} attempts, successfully placed {len(positions)}/{len(object_types)} objects.")
                    return positions
                else:
                    logger.debug(f"❌ Overall attempt #{overall_attempt}: Final verification failed, positions overlap.")
            
            # If failed, clear temporary data and retry
            positions.clear()
            placed_objects.clear()
        
        # All attempts failed
        logger.warning(f"⚠️ After {max_overall_attempts} overall attempts, could not find completely safe positions for all objects.")
        return []
    
    def _find_safe_position(self, object_type: str, existing_objects: List[Dict], 
                           max_attempts: int = 100) -> Optional[np.ndarray]:
        """
        Finds a safe position for a single object.
        
        Args:
            object_type: The type of the object.
            existing_objects: A list of already existing objects.
            max_attempts: The maximum number of attempts.
            
        Returns:
            A safe position, or None if not found.
        """
        object_size = self.object_sizes.get(object_type, self.object_sizes["default"])
        
        for attempt in range(max_attempts):
            # Generate a random position
            position = self._generate_random_position(object_type)
            
            # Check if it's safe
            if self._is_position_safe(position, object_type, existing_objects):
                return position
        
        logger.warning(f"⚠️ After {max_attempts} attempts, could not find a safe position for an object of type '{object_type}'.")
        return None
    
    def _generate_random_position(self, object_type: str) -> np.ndarray:
        """Generates a random position."""
        object_size = self.object_sizes.get(object_type, self.object_sizes["default"])
        
        # Plates have a special placement strategy: further from the robot and at a lower height
        if object_type == "plate":
            x_min = 0.20  # Min X for plate: 20cm (away from robot)
            x_max = 0.45  # Max X for plate: 45cm
            y_min = self.workspace_bounds["y"][0] + object_size["radius"] + self.min_distance_from_edge  
            y_max = self.workspace_bounds["y"][1] - object_size["radius"] - self.min_distance_from_edge
            z_min = 0.01  # Min height for plate: 1cm
            z_max = 0.03  # Max height for plate: 3cm
        else:
            # Other objects use the standard workspace
            x_min = self.workspace_bounds["x"][0] + object_size["radius"] + self.min_distance_from_edge
            x_max = self.workspace_bounds["x"][1] - object_size["radius"] - self.min_distance_from_edge
            y_min = self.workspace_bounds["y"][0] + object_size["radius"] + self.min_distance_from_edge  
            y_max = self.workspace_bounds["y"][1] - object_size["radius"] - self.min_distance_from_edge
            z_min = self.workspace_bounds["z"][0]
            z_max = self.workspace_bounds["z"][1]
        
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)
        
        return np.array([x, y, z])
    
    def _is_position_safe(self, position: np.ndarray, object_type: str, 
                         existing_objects: List[Dict]) -> bool:
        """
        Checks if a position is safe.
        
        Args:
            position: The position to check.
            object_type: The type of the object.
            existing_objects: A list of already existing objects.
            
        Returns:
            True if the position is safe, False otherwise.
        """
        object_size = self.object_sizes.get(object_type, self.object_sizes["default"])
        
        # 1. Check if it's within the workspace
        if not self._is_within_workspace(position, object_size):
            return False
        
        # 2. Check the distance from the robot
        if not self._is_far_from_robot(position, object_size):
            return False
        
        # 3. Check the distance from other objects
        for existing_obj in existing_objects:
            if not self._is_far_from_object(position, object_size, existing_obj):
                return False
        
        return True
    
    def _is_within_workspace(self, position: np.ndarray, object_size: Dict) -> bool:
        """Checks if the object is within the workspace boundaries."""
        x, y, z = position
        radius = object_size["radius"]
        
        # Check X-axis
        if x - radius < self.workspace_bounds["x"][0] or x + radius > self.workspace_bounds["x"][1]:
            return False
        
        # Check Y-axis
        if y - radius < self.workspace_bounds["y"][0] or y + radius > self.workspace_bounds["y"][1]:
            return False
        
        # Check Z-axis
        if z < self.workspace_bounds["z"][0] or z > self.workspace_bounds["z"][1]:
            return False
        
        return True
    
    def _is_far_from_robot(self, position: np.ndarray, object_size: Dict) -> bool:
        """Checks if the object is outside the robot's exclusion zone."""
        x, y, z = position[0], position[1], position[2]
        radius = object_size["radius"]
        
        # Check if the object overlaps with the robot's exclusion zone
        x_min, x_max = self.robot_exclusion_zone["x"]
        y_min, y_max = self.robot_exclusion_zone["y"]
        z_min, z_max = self.robot_exclusion_zone["z"]
        
        # Expand the exclusion zone by the object's radius for the check
        if (x - radius < x_max and x + radius > x_min and
            y - radius < y_max and y + radius > y_min and
            z - radius < z_max and z + radius > z_min):
            return False  # Unsafe: Inside the exclusion zone
            
        return True  # Safe: Outside the exclusion zone
    
    def _is_far_from_object(self, position: np.ndarray, object_size: Dict, 
                           existing_obj: Dict) -> bool:
        """Checks the distance from another object."""
        existing_pos = existing_obj["position"]
        existing_type = existing_obj["type"]
        existing_size = self.object_sizes.get(existing_type, self.object_sizes["default"])
        
        # Calculate the distance in the XY plane only
        position_2d = position[:2]
        existing_pos_2d = existing_pos[:2]
        distance = np.linalg.norm(position_2d - existing_pos_2d)
        
        # The required minimum distance depends on the object type
        if existing_type == "plate":
            # For plates, use a dedicated distance setting (not based on radius sum)
            required_distance = self.min_distance_from_plate
        else:
            # For other objects, it's the sum of their radii plus a safety margin
            required_distance = (object_size["radius"] + existing_size["radius"] + 
                               self.min_distance_between_objects)
        
        is_safe = distance >= required_distance
        
        # Debug logging for distance checks
        if not is_safe:
            logger.debug(f"💥 Insufficient distance! Position {position[:2]} is {distance:.3f}m from {existing_obj['name']}({existing_type}) at {existing_pos[:2]}. Required: {required_distance:.3f}m")
        else:
            logger.debug(f"✅ Safe distance: Position {position[:2]} is {distance:.3f}m from {existing_obj['name']}({existing_type}) at {existing_pos[:2]}. Required: {required_distance:.3f}m")
        
        return is_safe
    
    def clear_placement_history(self):
        """Clears the placement history."""
        self.placed_objects = []
        logger.info("🔄 Smart placement history has been cleared.")
    
    def _verify_all_positions_safe(self, positions: List[np.ndarray], object_types: List[str]) -> bool:
        """
        Verifies that all generated positions are truly safe and non-overlapping.
        
        Args:
            positions: The list of positions to verify.
            object_types: The corresponding list of object types.
            
        Returns:
            True if all positions are safe.
        """
        # 1. Check for overlaps among the newly generated positions
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos1, pos2 = positions[i], positions[j]
                type1, type2 = object_types[i], object_types[j]
                
                size1 = self.object_sizes.get(type1, self.object_sizes["default"])
                size2 = self.object_sizes.get(type2, self.object_sizes["default"])
                
                # Calculate distance in the XY plane
                distance_xy = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                required_distance = size1["radius"] + size2["radius"] + self.min_distance_between_objects
                
                if distance_xy < required_distance:
                    logger.debug(f"❌ Position verification failed: {type1} and {type2} are {distance_xy:.3f}m apart, but require {required_distance:.3f}m.")
                    return False
        
        # 2. Check for overlaps between new positions and already existing objects
        for i, pos in enumerate(positions):
            obj_type = object_types[i]
            obj_size = self.object_sizes.get(obj_type, self.object_sizes["default"])
            
            for existing_obj in self.placed_objects:
                if not self._is_far_from_object(pos, obj_size, existing_obj):
                    logger.debug(f"❌ Position verification failed: {obj_type} overlaps with existing object {existing_obj['name']}.")
                    return False
        
        return True

    def get_placement_info(self) -> Dict:
        """Gets the current placement information."""
        return {
            "workspace_bounds": self.workspace_bounds,
            "placed_objects": self.placed_objects.copy(),
            "object_count": len(self.placed_objects)
        }
