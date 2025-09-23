# -*- coding: utf-8 -*-
"""
SingleJawGripper - Adapted from reference/so101_pickup_project/core_modules/single_gripper.py
This module contains a high-level controller for a single-jaw gripper, like the one on the SO-101 robot.
The code has been integrated into this project to remove dependencies on the 'reference' directory.
"""

from typing import Callable, List, Optional

import numpy as np
import omni.kit.app
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper


class SingleJawGripper(Gripper):
    """Provides high-level control functions for a single-acting jaw gripper (e.g., the SO-101 gripper).

    Args:
        end_effector_prim_path (str): The prim path of the gripper's fixed part.
        joint_prim_name (str): The name of the single active gripper joint to be controlled.
        joint_opened_position (float): The position of the joint in the "open" state (in radians).
        joint_closed_position (float): The position of the joint in the "closed" state (in radians).
        action_delta (float, optional): An incremental value to apply when opening or closing. If None, it sets the target position directly. Defaults to None.
    """

    def __init__(
        self,
        end_effector_prim_path: str,
        joint_prim_name: str,
        joint_opened_position: float,
        joint_closed_position: float,
        action_delta: Optional[float] = None,
    ) -> None:
        """
        Note: We intentionally do not call super().__init__(...).
        This is because the Gripper base class creates a RigidPrimView,
        which conflicts with ArticulationView during world.reset() and causes a physics crash.
        We treat this gripper as a purely logical controller.
        """
        self._joint_prim_name = joint_prim_name
        self._joint_opened_position = joint_opened_position
        self._joint_closed_position = joint_closed_position
        self._action_delta = action_delta

        # Explicitly initialize all instance variables in the constructor.
        self._articulation_num_dofs: Optional[int] = None
        self._articulation_apply_action_func: Optional[Callable[[ArticulationAction], None]] = None
        self._joint_dof_index: Optional[int] = None
        return

    def initialize(
        self,
        articulation_apply_action_func: Callable[[ArticulationAction], None],
        articulation_num_dofs: int,
        joint_dof_indicies: List[int],
    ) -> None:
        """Called by the SingleManipulator at the start of the simulation.

        Args:
            articulation_apply_action_func (Callable[[ArticulationAction], None]): The function used to apply actions to the entire robot arm.
            articulation_num_dofs (int): The total number of degrees of freedom (DOF) for the entire arm.
            joint_dof_indicies (List[int]): The index of the joint controlled by this gripper in the arm's joint list.
        """
        self._articulation_apply_action_func = articulation_apply_action_func
        self._articulation_num_dofs = articulation_num_dofs
        # For a single-joint gripper, we only care about the first index.
        self._joint_dof_index = joint_dof_indicies[0]

    def open(self) -> None:
        """Opens the gripper."""
        # Note: This is a blocking call.
        self._articulation_apply_action_func(self.forward(action="open"))

    def close(self) -> None:
        """Closes the gripper."""
        # Note: This is a blocking call.
        self._articulation_apply_action_func(self.forward(action="close"))

    def forward(self, action: str) -> ArticulationAction:
        """Calculates the joint action based on 'open' or 'close'."""
        target_joint_positions = [None] * self._articulation_num_dofs

        # Directly use the preset target positions.
        if action == "open":
            target_joint_positions[self._joint_dof_index] = self._joint_opened_position
        elif action == "close":
            target_joint_positions[self._joint_dof_index] = self._joint_closed_position
        else:
            raise ValueError(f"Action '{action}' is not defined for SingleJawGripper.")

        return ArticulationAction(joint_positions=target_joint_positions)

    def apply_action(self, control_actions: ArticulationAction) -> None:
        """
        Applies actions to the articulation that this gripper belongs to.

        Args:
            control_actions (ArticulationAction): actions to be applied.
        """
        self._articulation_apply_action_func(control_actions)
        return
