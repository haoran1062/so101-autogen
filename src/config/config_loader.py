# -*- coding: utf-8 -*-
"""
Configuration Loader - Corresponds to the parameter management of the main script.
Responsible for loading and managing configuration files, providing an interface
compatible with the original script's args_cli.
"""

import yaml
import os
import argparse
from typing import Dict, Any
import logging

class ConfigLoader:
    """Configuration loader that provides an interface compatible with the main script's args_cli."""
    
    def __init__(self, config_path: str = None):
        """
        Initializes the ConfigLoader.
        
        Args:
            config_path: Path to the configuration file. Defaults to config/scene_config.yaml in the project root.
        """
        if config_path is None:
            # Default config file path - points to the config folder in the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to the project root
            config_path = os.path.join(project_root, "config", "scene_config.yaml")
        
        self.config_path = config_path
        self.config = {}
        self.args_cli = None  # Object compatible with the original script's args_cli
        
        # Load the configuration file
        self.load_config()
        
        # Create the compatible args_cli object
        self.create_args_cli_compatible()
    
    def load_config(self):
        """Loads the YAML configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            
            print(f"✅ Configuration file loaded successfully: {self.config_path}")
            
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"❌ Error in configuration file format: {e}")
            raise
    
    def get_config(self) -> dict:
        """Gets the configuration dictionary."""
        return self.config
    
    def create_args_cli_compatible(self):
        """Creates an object compatible with the main script's args_cli."""
        # Create a mock argparse.Namespace object
        self.args_cli = argparse.Namespace()
        
        # Map from the configuration file to args_cli attributes
        # Corresponds to parameter definitions in the main script, lines 115-128
        
        # Simulation parameters
        self.args_cli.headless = self.config.get('simulation', {}).get('headless', False)
        
        # Scene parameters
        plate_config = self.config.get('scene', {}).get('plate', {})
        self.args_cli.plate_pos = plate_config.get('position', [0.25, -0.15, 0.1])
        
        oranges_config = self.config.get('scene', {}).get('oranges', {})
        self.args_cli.num_oranges = oranges_config.get('count', 3)
        
        robot_config = self.config.get('robot', {})
        self.args_cli.target_pos = robot_config.get('target_position', [0.3, 0.0, 0.15])
        
        # Task parameters
        task_config = self.config.get('task', {})
        debug_config = self.config.get('debug', {})
        
        self.args_cli.log_level = debug_config.get('log_level', 'INFO')
        self.args_cli.shoulder_pan_limit = task_config.get('shoulder_pan_limit', 110.0)
        self.args_cli.auto = False  # Default to non-automatic mode
        
        # Feature switches
        cameras_config = self.config.get('cameras', {})
        data_config = self.config.get('data_collection', {})
        
        self.args_cli.save_camera_data = cameras_config.get('save_data', False)
        self.args_cli.enable_data_collection = data_config.get('enabled', False)
        self.args_cli.data_output = data_config.get('output_path', './datasets/so101_pickup_data.hdf5')
        self.args_cli.no_search_mode = not task_config.get('search_mode', True)  # Note the logical inversion
        self.args_cli.show_debug_viz = debug_config.get('show_viz', False)
        
        print("✅ args_cli compatible object created successfully.")
        print(f"   headless: {self.args_cli.headless}")
        print(f"   plate_pos: {self.args_cli.plate_pos}")
        print(f"   num_oranges: {self.args_cli.num_oranges}")
        print(f"   target_pos: {self.args_cli.target_pos}")
        print(f"   no_search_mode: {self.args_cli.no_search_mode}")
        print(f"   show_debug_viz: {self.args_cli.show_debug_viz}")
    
    def get_config(self, section: str = None) -> Dict[str, Any]:
        """
        Gets the configuration content.
        
        Args:
            section: The name of the configuration section. If None, returns the entire configuration.
            
        Returns:
            A configuration dictionary.
        """
        if section is None:
            return self.config
        else:
            return self.config.get(section, {})
    
    def get_args_cli(self):
        """Gets the compatible args_cli object."""
        return self.args_cli
    
    def get_orange_generation_config(self) -> Dict[str, Any]:
        """Gets the orange generation config, corresponding to ORANGE_GENERATION_CONFIG in the main script."""
        return self.config.get('scene', {}).get('oranges', {}).get('generation', {})
    
    def get_grasp_detection_config(self) -> Dict[str, Any]:
        """Gets the grasp detection config, corresponding to GRASP_DETECTION_CONFIG in the main script."""
        return self.config.get('grasp_detection', {})
    
    def update_from_command_line(self, args):
        """Updates the configuration from command-line arguments."""
        if hasattr(args, 'headless') and args.headless is not None:
            self.args_cli.headless = args.headless
            
        if hasattr(args, 'plate_pos') and args.plate_pos is not None:
            self.args_cli.plate_pos = args.plate_pos
            
        if hasattr(args, 'num_oranges') and args.num_oranges is not None:
            self.args_cli.num_oranges = args.num_oranges
            
        # More command-line override logic can be added as needed
        print("✅ Configuration updated from command-line arguments.")

# Global instance of the config loader
_config_loader = None

def get_config_loader(config_path: str = None) -> ConfigLoader:
    """Gets the global instance of the config loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader

def get_args_cli():
    """Gets the compatible args_cli object for use by other modules."""
    return get_config_loader().get_args_cli()

if __name__ == "__main__":
    # Test the ConfigLoader
    loader = ConfigLoader()
    print("Configuration file content:")
    print(yaml.dump(loader.get_config(), default_flow_style=False, allow_unicode=True))
    
    print("\nargs_cli compatible object:")
    args_cli = loader.get_args_cli()
    for attr in dir(args_cli):
        if not attr.startswith('_'):
            print(f"  {attr}: {getattr(args_cli, attr)}")
