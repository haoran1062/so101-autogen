# -*- coding: utf-8 -*-
"""
Configuration Utilities Module
Provides functions for loading and processing configuration parameters, extracted from the main script.
"""

import os
import yaml


class ConfigManager:
    """Configuration Manager"""
    
    def __init__(self, project_root):
        """Initializes the ConfigManager.
        
        Args:
            project_root (str): The root path of the project.
        """
        self.project_root = project_root
        self.config_path = os.path.join(project_root, "config", "scene_config.yaml")
    
    def load_scene_config(self):
        """Loads the scene configuration file.
        
        Returns:
            dict: A dictionary with the scene configuration, or None if the file does not exist.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Scene configuration file loaded: {self.config_path}")
            return config
        else:
            print(f"⚠️ Configuration file not found: {self.config_path}")
            return None
    
    def get_config_with_defaults(self, config, key_path, default_value):
        """Safely retrieves a value from a nested configuration, using a default if not found.
        
        Args:
            config (dict): The configuration dictionary.
            key_path (str): The path to the key, using dot notation (e.g., "scene.plate.position").
            default_value: The default value to return if the key is not found.
            
        Returns:
            The configuration value or the default value.
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                print(f"⚠️ Configuration path '{key_path}' not found. Using default value: {default_value}")
                return default_value
        
        return current
    
    def get_plate_config(self, config):
        """Gets the plate configuration parameters.
        
        Args:
            config (dict): The scene configuration.
            
        Returns:
            dict: The plate configuration parameters.
        """
        return {
            "position": self.get_config_with_defaults(config, "scene.plate.position", [0.28, 0.0, 0.1]),
            "radius": self.get_config_with_defaults(config, "scene.plate.virtual_config.radius", 0.1),
            "height": self.get_config_with_defaults(config, "scene.plate.virtual_config.height", 0.02),
            "scale": self.get_config_with_defaults(config, "scene.plate.scale", 1.0)
        }
    
    def get_orange_config(self, config):
        """Gets the orange configuration parameters.
        
        Args:
            config (dict): The scene configuration.
            
        Returns:
            dict: The orange configuration parameters.
        """
        orange_generation = self.get_config_with_defaults(config, "scene.oranges.generation", {})
        
        return {
            "count": self.get_config_with_defaults(config, "scene.oranges.count", 3),
            "mass": self.get_config_with_defaults(config, "scene.oranges.physics.mass", 0.15),
            "models": self.get_config_with_defaults(config, "scene.oranges.models", 
                ["Orange001", "Orange002", "Orange003"]),
            "usd_paths": self.get_config_with_defaults(config, "scene.oranges.usd_paths", [
                "assets/objects/Orange001/Orange001.usd",
                "assets/objects/Orange002/Orange002.usd", 
                "assets/objects/Orange003/Orange003.usd"
            ]),
            "x_range": orange_generation.get("x_range", [0.1, 0.2]),
            "y_range": orange_generation.get("y_range", [0.03, 0.23]),
            "z_drop_height": orange_generation.get("z_drop_height", 0.1),
            "orange_radius": orange_generation.get("orange_radius", 0.025),
            "min_distance": orange_generation.get("min_distance", 0.06),
            "max_attempts": orange_generation.get("max_attempts", 50)
        }
    
    def get_target_configs(self, config):
        """Gets the target configuration parameters.
        
        Args:
            config (dict): The scene configuration.
            
        Returns:
            dict: The target configuration parameters.
        """
        return self.get_config_with_defaults(config, "target_configs", {
            "/World/orange1": {
                "name": "orange1_object",
                "draw_aabb": True,
                "aabb_color": (1.0, 1.0, 0.0, 0.5),
                "draw_obb": True,
                "obb_color": (0.0, 1.0, 1.0, 1.0),
            },
            "/World/orange2": {
                "name": "orange2_object",
                "draw_aabb": True,
                "aabb_color": (1.0, 1.0, 0.0, 0.5),
                "draw_obb": True,
                "obb_color": (0.0, 1.0, 1.0, 1.0),
            },
            "/World/orange3": {
                "name": "orange3_object",
                "draw_aabb": True,
                "aabb_color": (1.0, 1.0, 0.0, 0.5),
                "draw_obb": True,
                "obb_color": (0.0, 1.0, 1.0, 1.0),
            }
        })


# Compatibility functions to maintain the same interface as the main script.
def load_scene_config(project_root):
    """Compatibility function."""
    config_manager = ConfigManager(project_root)
    return config_manager.load_scene_config()


def get_config_with_defaults(config, key_path, default_value):
    """Compatibility function."""
    config_manager = ConfigManager("")  # Empty project root, only used for method access.
    return config_manager.get_config_with_defaults(config, key_path, default_value)
