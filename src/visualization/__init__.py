# -*- coding: utf-8 -*-
"""
Visualization Module - Contains functionalities for bounding boxes, rays, debug visualization, etc.
"""

# Lazy imports to avoid import errors in non-Isaac Sim environments
def get_bbox_visualizer():
    """Lazily imports the BoundingBoxVisualizer."""
    from .bbox_visualizer import BoundingBoxVisualizer
    return BoundingBoxVisualizer

def get_pickup_assessor():
    """Lazily imports the PickupAssessor."""
    from .pickup_assessor import PickupAssessor
    return PickupAssessor

def get_ray_visualizer():
    """Lazily imports the RayVisualizer."""
    from .ray_visualizer import RayVisualizer
    return RayVisualizer

def get_debug_visualizer():
    """Lazily imports the DebugVisualizer manager."""
    from .debug_visualizer import DebugVisualizer
    return DebugVisualizer

__all__ = [
    'get_bbox_visualizer', 
    'get_pickup_assessor', 
    'get_ray_visualizer', 
    'get_debug_visualizer'
]

