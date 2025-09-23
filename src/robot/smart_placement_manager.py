"""
Smart Placement Manager
Dynamically calculates the optimal placement position for objects on a plate to avoid collisions.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PlacedObject:
    """Information about a placed object."""
    name: str
    position: np.ndarray  # [x, y, z]
    radius: float
    height: float
    success: bool = True

class SmartPlacementManager:
    """Smart Placement Manager"""
    
    def __init__(self, config: Dict = None):
        """
        Initializes the Smart Placement Manager.
        
        Args:
            config: A configuration dictionary containing plate information and placement parameters.
        """
        # Default configuration
        self.config = {
            # Plate configuration
            "plate_radius": 0.10,  # 10cm plate radius
            "plate_height": 0.02,  # 2cm plate height
            "plate_position": [0.28, -0.05, 0.02],  # Plate position
            
            # Object configuration
            "object_radius": 0.025,  # 2.5cm orange radius
            "object_height": 0.05,   # 5cm orange height
            
            # Safety margins
            "safety_margin_edge": 0.025,  # 2.5cm safety margin from the plate edge
            "safety_margin_objects": 0.06,  # 6cm minimum distance between objects
            
            # Placement range limits (to avoid unreachable IK targets)
            "max_x_distance": 0.30,  # 30cm max distance in X-direction
            "max_y_distance": 0.20,  # 20cm max distance in Y-direction
            
            # Placement strategy
            "placement_strategy": "center_to_edge",  # From center to edge
            "max_placement_attempts": 100,  # Maximum number of placement attempts
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Record of placed objects
        self.placed_objects: List[PlacedObject] = []
        
        # Plate information
        self.plate_center = np.array(self.config["plate_position"][:2])  # XY only
        self.plate_radius = self.config["plate_radius"]
        self.effective_radius = self.plate_radius - self.config["safety_margin_edge"]
        
        logger.info(f"🎯 Smart Placement Manager initialized.")
        logger.info(f"    Plate Center: {self.plate_center}")
        logger.info(f"    Plate Radius: {self.plate_radius*100:.1f}cm")
        logger.info(f"    Effective Radius: {self.effective_radius*100:.1f}cm")
        logger.info(f"    Reachable Range: X≤{self.config['max_x_distance']*100:.1f}cm, Y≤{self.config['max_y_distance']*100:.1f}cm")
    
    def calculate_placement_position(self, target_object_name: str) -> Optional[np.ndarray]:
        """
        Calculates the optimal placement position.
        
        Args:
            target_object_name: The name of the target object.
            
        Returns:
            The optimal placement position [x, y, z], or None if no suitable position is found.
        """
        try:
            logger.info(f"🎯 Calculating placement position for {target_object_name}")
            logger.info(f"    Number of currently placed objects: {len(self.placed_objects)}")
            
            # Generate candidate positions
            candidate_positions = self._generate_candidate_positions()
            
            if not candidate_positions:
                logger.warning("⚠️ No available candidate positions.")
                return None
            
            # Select the best position
            best_position = self._select_best_position(candidate_positions, target_object_name)
            
            if best_position is not None:
                # Add the Z-coordinate (plate height + half of object height)
                z_position = self.config["plate_position"][2] + self.config["object_height"] / 2
                final_position = np.array([best_position[0], best_position[1], z_position])
                
                logger.info(f"✅ Optimal placement position calculated: [{final_position[0]:.4f}, {final_position[1]:.4f}, {final_position[2]:.4f}]")
                return final_position
            else:
                logger.warning("⚠️ Could not find a suitable placement position.")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate placement position: {e}")
            return None
    
    def _generate_candidate_positions(self) -> List[np.ndarray]:
        """Generates candidate placement positions."""
        try:
            candidate_positions = []
            
            # Strategy depends on the number of placed objects
            placed_count = len(self.placed_objects)
            
            if placed_count == 0:
                # First orange: Choose a position far from the origin (on the plate edge)
                logger.debug("🔍 First orange: selecting an edge position far from the origin.")
                edge_positions = self._generate_edge_positions()
                candidate_positions.extend(edge_positions)
                
            elif placed_count == 1:
                # Second orange: Choose a middle position, avoiding collision with the first
                logger.debug("🔍 Second orange: selecting a middle position.")
                middle_positions = self._generate_middle_positions()
                candidate_positions.extend(middle_positions)
                
            else:
                # Third and subsequent oranges: Find the most suitable open spot
                logger.debug("🔍 Third and subsequent: finding an open spot.")
                all_positions = self._generate_all_positions()
                candidate_positions.extend(all_positions)
            
            # Filter out positions outside the plate's effective radius
            valid_positions = []
            for pos in candidate_positions:
                distance_from_center = np.linalg.norm(pos - self.plate_center)
                if distance_from_center <= self.effective_radius:
                    valid_positions.append(pos)
            
            logger.debug(f"🔍 Generated {len(valid_positions)} valid candidate positions.")
            return valid_positions
            
        except Exception as e:
            logger.error(f"❌ Failed to generate candidate positions: {e}")
            return []
    
    def _generate_edge_positions(self) -> List[np.ndarray]:
        """Generates edge positions (far from the origin, but within a reasonable range)."""
        positions = []
        
        # Calculate the direction away from the origin (from origin towards plate center)
        origin = np.array([0.0, 0.0])
        direction = self.plate_center - origin
        direction = direction / np.linalg.norm(direction)  # Normalize
        
        # Generate positions in the direction away from the origin, but limited to a max distance
        for distance_ratio in [0.5, 0.6, 0.7]:  # Reduced distance ratios to avoid being too far
            pos = self.plate_center + direction * self.effective_radius * distance_ratio
            
            # Check if within a reasonable range
            if self._is_position_in_reachable_range(pos):
                positions.append(pos)
        
        # Also generate some positions in the perpendicular direction
        perpendicular = np.array([-direction[1], direction[0]])  # Perpendicular vector
        for distance_ratio in [0.4, 0.5, 0.6]:  # Reduced distance ratios
            pos = self.plate_center + perpendicular * self.effective_radius * distance_ratio
            if self._is_position_in_reachable_range(pos):
                positions.append(pos)
            
            pos = self.plate_center - perpendicular * self.effective_radius * distance_ratio
            if self._is_position_in_reachable_range(pos):
                positions.append(pos)
        
        return positions
    
    def _is_position_in_reachable_range(self, position: np.ndarray) -> bool:
        """Checks if a position is within the robot arm's reachable range."""
        try:
            # Check X-direction distance
            if abs(position[0]) > self.config["max_x_distance"]:
                return False
            
            # Check Y-direction distance
            if abs(position[1]) > self.config["max_y_distance"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to check position reachability: {e}")
            return False
    
    def _generate_middle_positions(self) -> List[np.ndarray]:
        """Generates middle positions."""
        positions = []
        
        # Generate positions around the plate center
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):  # 8 directions
            for distance_ratio in [0.2, 0.3, 0.4]:  # Reduced distance ratios to avoid being too far
                x = self.plate_center[0] + self.effective_radius * distance_ratio * np.cos(angle)
                y = self.plate_center[1] + self.effective_radius * distance_ratio * np.sin(angle)
                pos = np.array([x, y])
                
                # Check if within reachable range
                if self._is_position_in_reachable_range(pos):
                    positions.append(pos)
        
        return positions
    
    def _generate_all_positions(self) -> List[np.ndarray]:
        """Generates all possible positions (for finding open spots)."""
        positions = []
        
        # Generate grid positions
        grid_size = 0.02  # 2cm grid
        x_range = np.arange(
            self.plate_center[0] - self.effective_radius,
            self.plate_center[0] + self.effective_radius + grid_size,
            grid_size
        )
        y_range = np.arange(
            self.plate_center[1] - self.effective_radius,
            self.plate_center[1] + self.effective_radius + grid_size,
            grid_size
        )
        
        for x in x_range:
            for y in y_range:
                pos = np.array([x, y])
                distance_from_center = np.linalg.norm(pos - self.plate_center)
                if distance_from_center <= self.effective_radius and self._is_position_in_reachable_range(pos):
                    positions.append(pos)
        
        return positions
    
    def _select_best_position(self, candidate_positions: List[np.ndarray], target_object_name: str) -> Optional[np.ndarray]:
        """Selects the best placement position."""
        try:
            if not candidate_positions:
                return None
            
            # If no objects are placed yet, choose the first position
            if not self.placed_objects:
                return candidate_positions[0]
            
            # Calculate a score for each candidate position
            best_position = None
            best_score = -1
            
            for pos in candidate_positions:
                score = self._calculate_position_score(pos)
                if score > best_score:
                    best_score = score
                    best_position = pos
            
            if best_position is not None:
                logger.debug(f"🔍 Selected position: [{best_position[0]:.4f}, {best_position[1]:.4f}], Score: {best_score:.2f}")
            
            return best_position
            
        except Exception as e:
            logger.error(f"❌ Failed to select best position: {e}")
            return None
    
    def _calculate_position_score(self, position: np.ndarray) -> float:
        """Calculates a score for a position (higher is better)."""
        try:
            score = 0.0
            
            # 1. Minimum distance to already placed objects (larger is better)
            min_distance = float('inf')
            for placed_obj in self.placed_objects:
                distance = np.linalg.norm(position - placed_obj.position[:2])
                min_distance = min(min_distance, distance)
            
            if min_distance < self.config["safety_margin_objects"]:
                return -1.0  # Too close, not usable
            
            # Distance score: further is better
            distance_score = min_distance / self.config["safety_margin_objects"]
            score += distance_score * 10  # Weight of 10
            
            # 2. Distance from the plate center (moderate is best)
            center_distance = np.linalg.norm(position - self.plate_center)
            center_score = 1.0 - abs(center_distance - self.effective_radius * 0.5) / (self.effective_radius * 0.5)
            score += center_score * 5  # Weight of 5
            
            # 3. Distance from the origin (further is better, to avoid arm collisions)
            origin_distance = np.linalg.norm(position)
            origin_score = origin_distance / (self.effective_radius * 2)
            score += origin_score * 3  # Weight of 3
            
            return score
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate position score: {e}")
            return -1.0
    
    def record_placement(self, object_name: str, position: np.ndarray, success: bool = True):
        """
        Records the result of an object placement.
        
        Args:
            object_name: The name of the object.
            position: The placement position.
            success: Whether the placement was successful.
        """
        try:
            placed_obj = PlacedObject(
                name=object_name,
                position=position,
                radius=self.config["object_radius"],
                height=self.config["object_height"],
                success=success
            )
            
            if success:
                self.placed_objects.append(placed_obj)
                logger.info(f"✅ Placement successful for: {object_name} at [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            else:
                logger.warning(f"⚠️ Placement failed for: {object_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to record placement result: {e}")
    
    def get_placement_summary(self) -> Dict:
        """Gets a summary of placement information."""
        return {
            "total_placed": len(self.placed_objects),
            "placed_objects": [
                {
                    "name": obj.name,
                    "position": obj.position.tolist(),
                    "success": obj.success
                }
                for obj in self.placed_objects
            ],
            "plate_info": {
                "center": self.plate_center.tolist(),
                "radius": self.plate_radius,
                "effective_radius": self.effective_radius
            }
        }
    
    def scan_existing_objects(self, world_scene, orange_names: List[str]):
        """
        Scans the scene for existing oranges and checks which are already on the plate.
        
        Args:
            world_scene: The world scene object.
            orange_names: A list of orange names.
        """
        try:
            logger.info("🔍 Scanning scene for existing oranges...")
            scanned_count = 0
            
            for orange_name in orange_names:
                try:
                    # Get the orange object
                    orange_object = world_scene.get_object(orange_name)
                    if orange_object is None:
                        continue
                    
                    # Get the orange's position
                    orange_pos, _ = orange_object.get_world_pose()
                    
                    # Check if it's within the plate's area
                    if self._is_object_in_plate(orange_pos):
                        # Check if it has already been recorded
                        if not any(obj.name == orange_name for obj in self.placed_objects):
                            placed_obj = PlacedObject(
                                name=orange_name,
                                position=orange_pos,
                                radius=self.config["object_radius"],
                                height=self.config["object_height"],
                                success=True
                            )
                            self.placed_objects.append(placed_obj)
                            scanned_count += 1
                            logger.info(f"✅ Found already placed orange: {orange_name} at [{orange_pos[0]:.4f}, {orange_pos[1]:.4f}, {orange_pos[2]:.4f}]")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to scan orange {orange_name}: {e}")
                    continue
            
            logger.info(f"🔍 Scene scan complete. Found {scanned_count} already placed oranges.")
            logger.info(f"    Total number of placed objects is now: {len(self.placed_objects)}")
            
            # Output the placement analysis
            self._print_placement_analysis()
            
        except Exception as e:
            logger.error(f"❌ Failed to scan scene: {e}")
    
    def _is_object_in_plate(self, object_pos: np.ndarray) -> bool:
        """Checks if an object is within the plate's area."""
        try:
            # Calculate the distance from the object to the plate's center
            distance_from_center = np.linalg.norm(object_pos[:2] - self.plate_center)
            
            # Check if within the plate's radius (with safety margin)
            return distance_from_center <= self.effective_radius
            
        except Exception as e:
            logger.error(f"❌ Failed to check if object is in plate: {e}")
            return False
    
    def _print_placement_analysis(self):
        """Prints the placement analysis results."""
        try:
            if not self.placed_objects:
                logger.info("📊 Placement Analysis: Plate is empty. Ready to place the first orange.")
                return
            
            logger.info("📊 Placement Analysis:")
            logger.info(f"    Number of placed oranges: {len(self.placed_objects)}")
            
            for i, obj in enumerate(self.placed_objects, 1):
                logger.info(f"    Orange {i}: {obj.name} at [{obj.position[0]:.4f}, {obj.position[1]:.4f}, {obj.position[2]:.4f}]")
            
            # Calculate available space
            available_space = self._calculate_available_space()
            logger.info(f"    Available Space Assessment: {available_space}")
            
        except Exception as e:
            logger.error(f"❌ Failed to print placement analysis: {e}")
    
    def _calculate_available_space(self) -> str:
        """Calculates an assessment of the available space."""
        try:
            placed_count = len(self.placed_objects)
            
            if placed_count == 0:
                return "Sufficient (space for 3 oranges)"
            elif placed_count == 1:
                return "Good (space for 2 more oranges)"
            elif placed_count == 2:
                return "Limited (space for 1 more orange)"
            else:
                return "Insufficient (placement is not recommended)"
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate available space: {e}")
            return "Unknown"
    
    def reset(self):
        """Resets the state of the placement manager, clearing all placed object records."""
        self.placed_objects.clear()
        logger.info("🔄 Smart Placement Manager state has been reset.")
        print("🔄 Smart Placement Manager state has been reset.")
