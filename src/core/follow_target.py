# -*- coding: utf-8 -*-
"""
FollowTarget Task - Adapted from reference/so101_pickup_project/core_modules/follow_target.py
This module contains a customized FollowTarget task for the SO-101 robot arm.
The code has been integrated into this project to remove dependencies on the 'reference' directory.
"""

import sys
import os
import numpy as np
from typing import Optional

# Isaac Sim imports
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.api.tasks as tasks

# Local imports for custom classes
from .single_gripper import SingleJawGripper
from .patched_manipulator import PatchedSingleManipulator


class FollowTarget(tasks.FollowTarget):
    """
    FollowTarget task for the SO-101 robot arm.
    This class is a direct adaptation from the original implementation in the reference project.
    """

    def __init__(
        self,
        name: str = "so101_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        """Sets up the SO-101 robot arm in the simulation."""
        # Get the project root to build absolute paths.
        # This uses the current project's path, not the reference path.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        # Use the correct USD path for the SO-101 robot.
        asset_path = os.path.join(project_root, "assets", "robots", "so101_physics_generated", "so101_physics.usd")
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/so101_robot")

        # Instantiate the custom single-jaw gripper.
        gripper = SingleJawGripper(
            end_effector_prim_path="/World/so101_robot/wrist_link",
            joint_prim_name="gripper",  # Use the joint name from the URDF.
            joint_opened_position=1.74533,
            joint_closed_position=0.0,
            action_delta=None,
        )

        # Instantiate the patched manipulator to handle the custom gripper.
        manipulator = PatchedSingleManipulator(
            prim_path="/World/so101_robot",
            name="so101_robot",  # Use a consistent robot name.
            end_effector_prim_path="/World/so101_robot/wrist_link",
            gripper=gripper
        )

        # Set the default state for the 6-DOF arm (excluding the gripper).
        joints_default_positions = np.zeros(6)
        manipulator.set_joints_default_state(positions=joints_default_positions)

        return manipulator
