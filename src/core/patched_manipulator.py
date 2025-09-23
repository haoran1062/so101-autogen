# -*- coding: utf-8 -*-
"""
PatchedSingleManipulator - Adapted from reference/so101_pickup_project/core_modules/patched_manipulator.py
This module provides a patched SingleManipulator to support the custom SingleJawGripper.
The code has been integrated into this project to remove dependencies on the 'reference' directory.
"""

import omni.physics.tensors
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims.impl.single_articulation import SingleArticulation
from isaacsim.robot.manipulators.manipulators import SingleManipulator

# Local import for SingleJawGripper
from .single_gripper import SingleJawGripper


class PatchedSingleManipulator(SingleManipulator):
    """A patched version of SingleManipulator to support the SingleJawGripper."""
    
    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None) -> None:
        super().initialize(physics_sim_view)
        if isinstance(self._gripper, SingleJawGripper):
            gripper_joint_name = self._gripper._joint_prim_name
            gripper_dof_index = self.get_dof_index(gripper_joint_name)
            self._gripper.initialize(
                articulation_apply_action_func=self.apply_action,
                articulation_num_dofs=self.num_dof,
                joint_dof_indicies=[gripper_dof_index],
            )
        return

    def post_reset(self) -> None:
        SingleArticulation.post_reset(self)
        return
