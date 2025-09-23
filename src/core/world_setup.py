# -*- coding: utf-8 -*-
"""
World Setup Manager
Handles the initialization of the Isaac Sim World, environment setup,
and the creation of simulation tasks.
"""

import numpy as np
from typing import Dict, Any
import logging

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
# Use a local copy of the FollowTarget task
from src.core.follow_target import FollowTarget
import omni.physx
from omni.isaac.core.utils.prims import create_prim

import omni.usd
from pxr import Sdf, Gf, UsdGeom

class WorldSetup:
    """
    World Setup Manager
    Responsible for creating the Isaac Sim world, adding tasks, and setting up the environment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the WorldSetup manager.
        
        Args:
            config: The configuration dictionary.
        """
        self.config = config
        self.world = None
        self.task = None
        
        # Get parameters from config
        sim_config = config.get('simulation', {})
        self.stage_units = sim_config.get('stage_units_in_meters', 1.0)
        
        robot_config = config.get('robot', {})
        self.target_position = np.array(robot_config.get('target_position', [0.3, 0.0, 0.15]))
        
        task_config = config.get('task', {})
        self.task_name = task_config.get('name', 'so101_follow_target')
    
    def create_world(self) -> World:
        """
        Creates the Isaac Sim world.
        """
        try:
            self.world = World(stage_units_in_meters=self.stage_units)
            
            print(f"✅ World created (stage_units_in_meters: {self.stage_units}).")
            return self.world
            
        except Exception as e:
            print(f"❌ Failed to create World: {e}")
            raise
    
    def setup_environment(self):
        """
        Sets up the environment by adding lighting and a ground plane.
        """
        try:
            # Get environment configuration
            env_config = self.config.get('scene', {}).get('environment', {})
            
            # 1. Add a dome light
            light_config = env_config.get('lighting', {}).get('dome_light', {})
            light_prim_path = light_config.get('path', '/World/defaultLight')
            
            create_prim(prim_path=light_prim_path, prim_type="DomeLight")
            
            stage = omni.usd.get_context().get_stage()
            light_prim = stage.GetPrimAtPath(light_prim_path)
            
            # Set light parameters
            intensity = light_config.get('intensity', 3000.0)
            color = light_config.get('color', [0.75, 0.75, 0.75])
            
            light_prim.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(intensity)
            light_prim.CreateAttribute("color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            
            print(f"✅ Dome light added: intensity={intensity}, color={color}.")
            
            # 2. Add a ground plane
            if env_config.get('ground_plane', True):
                self.world.scene.add_default_ground_plane()
                print("✅ Default ground plane added.")
            
        except Exception as e:
            print(f"❌ Failed to set up environment: {e}")
            raise
    
    def add_follow_target_task(self):
        """
        Adds the FollowTarget task to the world.
        """
        try:
            self.task = FollowTarget(name=self.task_name, target_position=self.target_position)
            
            self.world.add_task(self.task)
            
            print(f"✅ FollowTarget task added: name={self.task_name}, target_position={self.target_position}.")
            return self.task
            
        except Exception as e:
            print(f"❌ Failed to add FollowTarget task: {e}")
            raise
    
    def reset_world(self):
        """
        Resets the world.
        """
        try:
            if self.world is not None:
                self.world.reset()
                print("✅ World has been reset.")
            else:
                print("⚠️ World not created, cannot reset.")
                
        except Exception as e:
            print(f"❌ Failed to reset World: {e}")
            raise
    
    def get_world(self) -> World:
        """Gets the World object."""
        return self.world
    
    def get_task(self):
        """Gets the task object."""
        return self.task
    
    def get_robot(self):
        """
        Gets the robot object from the world scene.
        """
        try:
            if self.task is None:
                print("⚠️ Task not created, cannot get robot.")
                return None
            
            task_params = self.task.get_params()
            robot_name = task_params["robot_name"]["value"]
            robot = self.world.scene.get_object(robot_name)
            
            print(f"✅ Robot object '{robot_name}' acquired.")
            return robot
            
        except Exception as e:
            print(f"❌ Failed to acquire robot object: {e}")
            return None
    
    def play_world(self):
        """
        Starts the simulation.
        """
        try:
            if self.world is not None:
                self.world.play()
                print("✅ Simulation started.")
            else:
                print("⚠️ World not created, cannot start simulation.")
                
        except Exception as e:
            print(f"❌ Failed to start simulation: {e}")
            raise
    
    def step_world(self, render: bool = True):
        """
        Executes a single simulation step.
        """
        if self.world is not None:
            self.world.step(render=render)
        else:
            print("⚠️ World not created, cannot execute simulation step.")
    
    def is_playing(self) -> bool:
        """Checks if the simulation is playing."""
        if self.world is not None:
            return self.world.is_playing()
        return False
