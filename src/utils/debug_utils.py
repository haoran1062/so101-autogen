# -*- coding: utf-8 -*-
"""
Debug Information Printing Utility
Provides functions for printing debug information, extracted from the main script.
"""

import numpy as np


class DebugPrinter:
    """Debug Information Printer"""
    
    @staticmethod
    def print_initial_debug_info(plate_center, orange_positions):
        """Prints detailed debug information after initial generation.
        
        Args:
            plate_center (list): The center position of the plate.
            orange_positions (list): A list of orange positions.
        """
        print("\n" + "="*60)
        print("🔍 Initial Generation Debug Information")
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
        orange_names = ["orange1_object", "orange2_object", "orange3_object"]
        for i, name in enumerate(orange_names):
            if i < len(orange_positions):
                pos = orange_positions[i]
                print(f"   {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # 3. Overlap detection
        overlap_detected = False
        if plate_center and orange_positions:
            print(f"\n⚠️ Overlap Detection:")
            plate_radius = 0.10
            
            for i, name in enumerate(orange_names):
                if i < len(orange_positions):
                    pos = orange_positions[i]
                    
                    # Calculate distance in the XY plane
                    distance_xy = np.sqrt((pos[0] - plate_center[0])**2 + (pos[1] - plate_center[1])**2)
                    
                    # Check if it's within the plate's XY area
                    is_in_plate_xy = distance_xy <= plate_radius
                    if is_in_plate_xy:
                        overlap_detected = True
                    
                    print(f"   {name}:")
                    print(f"     Distance to plate center (XY): {distance_xy:.3f}m")
                    print(f"     Within plate's XY area: {'❌ Yes' if is_in_plate_xy else '✅ No'}")
        
        print("="*60 + "\n")
    
    @staticmethod
    def check_orange_plate_overlap(plate_center, orange_positions):
        """Checks if any orange overlaps with the plate.
        
        Args:
            plate_center (list): The center position of the plate.
            orange_positions (list): A list of orange positions.
            
        Returns:
            bool: True if there is an overlap, False otherwise.
        """
        if not plate_center or not orange_positions:
            return False
        
        plate_radius = 0.10  # 10cm radius
        
        for i, pos in enumerate(orange_positions):
            # Calculate distance in the XY plane
            distance_xy = np.sqrt((pos[0] - plate_center[0])**2 + (pos[1] - plate_center[1])**2)
            
            # Check if it's within the plate's XY area
            if distance_xy <= plate_radius:
                return True
        
        return False


# Compatibility functions to maintain the same interface as the main script.
def print_initial_debug_info(plate_center, orange_positions):
    """Compatibility function."""
    DebugPrinter.print_initial_debug_info(plate_center, orange_positions)


def check_orange_plate_overlap(plate_center, orange_positions):
    """Compatibility function."""
    return DebugPrinter.check_orange_plate_overlap(plate_center, orange_positions)
