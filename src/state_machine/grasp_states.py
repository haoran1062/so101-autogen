# -*- coding: utf-8 -*-
"""
Simplified Grasping State Enumeration
This defines a streamlined state machine, removing complex search, confirmation,
and recovery logic for a more direct grasping process.
"""

from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SimpleGraspingState(Enum):
    """
    Enumeration for the simplified grasping state machine.
    Defines a linear workflow without search, confirmation, or recovery states.
    """
    IDLE = "idle"                    # Idle - waiting for user to select a target
    APPROACH = "approach"            # Approach - move above the target
    POSTURE_ADJUST = "posture_adjust" # Posture Adjust - align the gripper to be vertical (descends directly on failure)
    DESCEND = "descend"              # Descend - lower to grasping height
    GRASP = "grasp"                  # Grasp - close the gripper
    LIFT = "lift"                    # Lift - lift the object
    RETREAT = "retreat"              # Retreat - move to a safe position
    TRANSPORT = "transport"          # Transport - move over the plate (with linear descent)
    RELEASE = "release"              # Release - open the gripper
    RETURN_HOME = "return_home"      # Return Home - move to the initial position
    SUCCESS = "success"              # Success - task completed successfully
    FAILED = "failed"                # Failed - task failed

    def __str__(self):
        return self.value

    def get_display_name(self):
        """Gets the display name for the state."""
        display_names = {
            self.IDLE: "Idle",
            self.APPROACH: "Approaching Target",
            self.POSTURE_ADJUST: "Adjusting Posture",
            self.DESCEND: "Descending to Grasp",
            self.GRASP: "Executing Grasp",
            self.LIFT: "Lifting Object",
            self.RETREAT: "Moving to Safe Position",
            self.TRANSPORT: "Transporting Object",
            self.RELEASE: "Releasing Object",
            self.RETURN_HOME: "Returning to Home",
            self.SUCCESS: "Task Successful",
            self.FAILED: "Task Failed"
        }
        return display_names.get(self, self.value)

    def is_active_state(self):
        """Checks if this is an active state (requiring robot action)."""
        active_states = {
            self.APPROACH, self.POSTURE_ADJUST, self.DESCEND,
            self.GRASP, self.LIFT, self.TRANSPORT, self.RELEASE, self.RETREAT
        }
        return self in active_states

    def is_terminal_state(self):
        """Checks if this is a terminal state."""
        return self in {self.SUCCESS, self.FAILED}

    def get_next_state_on_success(self):
        """Gets the next state upon successful completion of the current one."""
        state_transitions = {
            self.APPROACH: self.POSTURE_ADJUST,
            self.POSTURE_ADJUST: self.DESCEND,    # Success in posture adjustment leads to descent
            self.DESCEND: self.GRASP,
            self.GRASP: self.LIFT,
            self.LIFT: self.TRANSPORT,
            self.TRANSPORT: self.RELEASE,
            self.RELEASE: self.RETREAT,
            self.RETREAT: self.SUCCESS
        }
        return state_transitions.get(self, self.FAILED)

    def get_next_state_on_failure(self):
        """Gets the next state upon failure (simplified logic)."""
        # Simplified logic: posture adjustment failure proceeds to descent, other failures mark the task as failed.
        if self == self.POSTURE_ADJUST:
            return self.DESCEND
        else:
            return self.FAILED

    @classmethod
    def get_state_flow_description(cls):
        """Gets a description of the state machine's workflow."""
        return """
Simplified Grasping State Flow:
1. IDLE - Waits for a number key press to select a target.
2. APPROACH - Moves the gripper above the target's XY coordinates.
3. POSTURE_ADJUST - Adjusts the gripper to be vertical (proceeds to descend on failure).
4. DESCEND - Lowers the gripper to the grasping height.
5. GRASP - Closes the gripper (marks as failed on failure).
6. LIFT - Lifts the object.
7. TRANSPORT - Moves the gripper above the plate.
8. RELEASE - Opens the gripper to release the object.
9. RETREAT - Moves back to a safe position.
10. SUCCESS/FAILED - Completes the task and returns to the IDLE state.

Key Characteristics:
- No search functionality.
- No user confirmation steps.
- Posture adjustment failure proceeds directly to the descent phase.
- Grasp failures are not retried.
- IK failures are reported and the state returns to idle.
"""

