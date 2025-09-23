# -*- coding: utf-8 -*-
"""
Random Position Generator
Supports orange position generation with dual-zone avoidance:
1.  Avoids the robot arm's central area.
2.  Avoids a circular safety zone around the plate.

This module does not depend on Isaac Sim and can be used independently.
"""

import random
import math
import logging
import numpy as np
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class RandomPositionGenerator:
    """
    Random Position Generator
    Supports dual-zone avoidance for the robot arm and the plate.
    """
    
    def __init__(self, config: Dict[str, Any], plate_position: List[float] = None):
        """
        Initializes the random position generator.
        
        Args:
            config: The configuration for orange generation.
            plate_position: The position of the plate, used for calculating the exclusion zone.
        """
        self.config = config
        
        # Base generation range
        self.x_range = tuple(config.get('x_range', [0.1, 0.35]))
        self.y_range = tuple(config.get('y_range', [-0.25, 0.25]))
        self.z_drop_height = config.get('z_drop_height', 0.1)
        self.max_attempts = config.get('max_attempts', 50)
        
        # Read physics properties from config
        physics_config = config.get('physics', {})
        self.orange_radius = physics_config.get('radius', 0.025)
        self.min_distance = physics_config.get('min_distance', 0.06)
        
        # Exclusion zone configuration
        self.exclusion_zones = config.get('exclusion_zones', [])
        self.plate_position = plate_position
        
        logger.info(f"✅ Random position generator initialized (dual-zone avoidance).")
        logger.info(f"   Base Range: X{self.x_range}, Y{self.y_range}, Z={self.z_drop_height}")
        logger.info(f"   Orange Radius: {self.orange_radius*1000:.1f}mm, Min Distance: {self.min_distance*1000:.1f}mm")
        logger.info(f"   Exclusion Zones: {len(self.exclusion_zones)}")
        if self.plate_position:
            logger.info(f"   Plate Position: {self.plate_position}")
    
    def set_plate_position(self, plate_position: List[float]):
        """Sets the plate position for dynamic avoidance calculation."""
        self.plate_position = plate_position
        logger.info(f"📍 Plate position has been updated: {self.plate_position}")
    
    def is_position_in_exclusion_zone(self, position: List[float]) -> bool:
        """
        Checks if a position falls within any exclusion zone.
        
        Args:
            position: The position [x, y, z] to check.
            
        Returns:
            True if the position is within an exclusion zone (invalid).
        """
        x, y, z = position
        
        for zone in self.exclusion_zones:
            zone_name = zone.get('name', 'unknown')
            zone_type = zone.get('type', 'rectangle')
            
            if zone_type == 'rectangle':
                # Rectangular exclusion zone (e.g., for the robot arm)
                bounds = zone.get('bounds', {})
                x_bounds = bounds.get('x', [-float('inf'), float('inf')])
                y_bounds = bounds.get('y', [-float('inf'), float('inf')])
                z_bounds = bounds.get('z', [-float('inf'), float('inf')])
                
                # Expand boundaries by the orange radius for the check
                if (x_bounds[0] - self.orange_radius <= x <= x_bounds[1] + self.orange_radius and
                    y_bounds[0] - self.orange_radius <= y <= y_bounds[1] + self.orange_radius and
                    z_bounds[0] - self.orange_radius <= z <= z_bounds[1] + self.orange_radius):
                    logger.debug(f"💥 Position [{x:.3f}, {y:.3f}, {z:.3f}] is within exclusion zone '{zone_name}'.")
                    return True
                    
            elif zone_type == 'circle':
                # Circular exclusion zone (e.g., for the plate)
                center = self._get_zone_center(zone)
                if center is None:
                    continue
                    
                radius = zone.get('radius', 0.15)
                z_bounds = zone.get('z', [0.0, 0.2])
                
                # Calculate distance in the XY plane
                distance_xy = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                
                # Check if it's within the circular zone (considering orange radius)
                if (distance_xy <= radius + self.orange_radius and
                    z_bounds[0] - self.orange_radius <= z <= z_bounds[1] + self.orange_radius):
                    logger.debug(f"💥 Position [{x:.3f}, {y:.3f}, {z:.3f}] is within exclusion zone '{zone_name}' (distance={distance_xy*1000:.1f}mm).")
                    return True
        
        return False
    
    def _get_zone_center(self, zone: Dict) -> List[float]:
        """Gets the center position of an exclusion zone."""
        if 'center' in zone:
            return zone['center']
        elif zone.get('center_from') == 'plate_position' and self.plate_position:
            return self.plate_position[:2]  # Use XY coordinates only
        else:
            logger.warning(f"⚠️ Could not determine the center for exclusion zone '{zone.get('name')}'.")
            return None
    
    def calculate_distance_2d(self, pos1: List[float], pos2: List[float]) -> float:
        """
        Calculates the 2D distance between two positions (ignoring the Z-axis).
        This function is kept for compatibility with the original script's logic.
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_position_valid(self, new_pos: List[float], existing_positions: List[List[float]], 
                         min_distance: float) -> bool:
        """
        Checks if a new position is valid (dual check).
        1. Not inside an exclusion zone.
        2. Maintains a minimum distance from existing positions.
        """
        # Check 1: Is it inside an exclusion zone?
        if self.is_position_in_exclusion_zone(new_pos):
            return False
        
        # Check 2: Is it far enough from existing positions?
        for existing_pos in existing_positions:
            if self.calculate_distance_2d(new_pos, existing_pos) < min_distance:
                return False
        
        return True
    
    def generate_random_orange_positions(self, num_oranges: int = 3) -> List[List[float]]:
        """
        Generates random, non-overlapping positions for oranges within a specified area.
        This version includes dual-zone avoidance.
        1. Avoids the robot arm area.
        2. Avoids the plate area.
        3. Ensures a minimum distance between oranges.
        """
        positions = []
        
        logger.info(f"🎲 Starting to generate {num_oranges} orange positions (dual-zone avoidance)...")
        logger.info(f"   Number of exclusion zones: {len(self.exclusion_zones)}")
        
        for i in range(num_oranges):
            attempts = 0
            position_found = False
            
            while attempts < self.max_attempts and not position_found:
                x = random.uniform(self.x_range[0], self.x_range[1])
                y = random.uniform(self.y_range[0], self.y_range[1])
                z = self.z_drop_height
                
                new_pos = [x, y, z]
                
                if self.is_position_valid(new_pos, positions, self.min_distance):
                    positions.append(new_pos)
                    position_found = True
                    logger.info(f"✅ Successfully generated position for orange #{i+1}: [{x:.3f}, {y:.3f}, {z:.3f}]")
                
                attempts += 1
            
            if not position_found:
                # Smart fallback position strategy
                fallback_pos = self._generate_fallback_position(i, positions)
                if fallback_pos:
                    positions.append(fallback_pos)
                    logger.warning(f"⚠️ Using fallback position for orange #{i+1}: [{fallback_pos[0]:.3f}, {fallback_pos[1]:.3f}, {fallback_pos[2]:.3f}]")
                else:
                    logger.error(f"❌ Could not generate any valid position for orange #{i+1}.")
        
        logger.info(f"🍊 Successfully generated {len(positions)} orange positions with a min distance of {self.min_distance*100:.1f}cm.")
        return positions
    
    def _generate_fallback_position(self, orange_index: int, existing_positions: List[List[float]]) -> List[float]:
        """
        Generates a smart fallback position by trying to place it in a safe zone.
        """
        # Define a few preset safe zones
        safe_zones = [
            # Front-left safe zone (avoids robot's Y-area)
            {"x": [0.12, 0.20], "y": [0.08, 0.20]},
            # Front-right safe zone (avoids robot's Y-area)
            {"x": [0.12, 0.20], "y": [-0.20, -0.08]},
            # Far-end safe zone (away from the plate)
            {"x": [0.30, 0.35], "y": [-0.15, 0.15]},
        ]
        
        for zone in safe_zones:
            for attempt in range(20):  # Try 20 times in each safe zone
                x = random.uniform(zone["x"][0], zone["x"][1])
                y = random.uniform(zone["y"][0], zone["y"][1])
                z = self.z_drop_height
                
                fallback_pos = [x, y, z]
                
                if self.is_position_valid(fallback_pos, existing_positions, self.min_distance):
                    logger.info(f"💡 Found a fallback position in a safe zone: [{x:.3f}, {y:.3f}, {z:.3f}]")
                    return fallback_pos
        
        # If all safe zones fail, use a simple linear arrangement
        fallback_x = self.x_range[0] + (orange_index * 0.08)
        fallback_y = self.y_range[0] + (orange_index * 0.06)
        return [fallback_x, fallback_y, self.z_drop_height]
    
    def get_config(self) -> Dict[str, Any]:
        """Gets the current configuration."""
        return {
            'x_range': self.x_range,
            'y_range': self.y_range,
            'z_drop_height': self.z_drop_height,
            'orange_radius': self.orange_radius,
            'min_distance': self.min_distance,
            'max_attempts': self.max_attempts
        }

# Utility functions, kept for compatibility with the original script.
def calculate_distance_2d(pos1: List[float], pos2: List[float]) -> float:
    """
    Calculates the 2D distance between two positions (ignoring the Z-axis).
    This function is kept for backward compatibility.
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def is_position_valid(new_pos: List[float], existing_positions: List[List[float]], 
                     min_distance: float) -> bool:
    """
    Checks if a new position maintains a minimum distance from existing positions.
    This function is kept for backward compatibility.
    """
    for existing_pos in existing_positions:
        if calculate_distance_2d(new_pos, existing_pos) < min_distance:
            return False
    return True

def generate_random_orange_positions(num_oranges: int = 3, 
                                   config: Dict[str, Any] = None) -> List[List[float]]:
    """
    Generates random, non-overlapping positions for oranges within a specified area.
    This function is kept for backward compatibility.
    """
    if config is None:
        # Use default config
        config = {
            "x_range": (0.1, 0.2),
            "y_range": (0.03, 0.23),
            "z_drop_height": 0.1,
            "orange_radius": 0.025,
            "min_distance": 0.06,
            "max_attempts": 50,
        }
    
    generator = RandomPositionGenerator(config)
    return generator.generate_random_orange_positions(num_oranges)

if __name__ == "__main__":
    # Test the random position generator
    test_config = {
        "x_range": (0.1, 0.2),
        "y_range": (0.03, 0.23),
        "z_drop_height": 0.1,
        "orange_radius": 0.025,
        "min_distance": 0.06,
        "max_attempts": 50,
    }
    
    generator = RandomPositionGenerator(test_config)
    
    # Generate a few test sets of positions
    for i in range(3):
        print(f"\nTest Run #{i+1}:")
        positions = generator.generate_random_orange_positions(3)
        
        # Verify distance constraints
        print("Verifying distance constraints:")
        for j, pos1 in enumerate(positions):
            for k, pos2 in enumerate(positions):
                if j < k:
                    distance = generator.calculate_distance_2d(pos1, pos2)
                    print(f"  Orange #{j+1} <-> Orange #{k+1}: {distance*100:.1f}cm (min required: {generator.min_distance*100:.1f}cm)")
    
    print("\nRandom position generator test complete.")
