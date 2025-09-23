# -*- coding: utf-8 -*-
"""
Keyboard Input Handler
"""

import logging
import weakref

# Isaac Sim imports (need to be imported after the simulation has started)
import carb.input
import omni.appwindow

logger = logging.getLogger(__name__)

class KeyboardHandler:
    """Handles keyboard input events."""
    
    def __init__(self, gripper_controller=None):
        """
        Initializes the keyboard handler.
        
        Args:
            gripper_controller: The gripper controller (optional).
        """
        self.gripper_ctrl = gripper_controller
        self.user_choice = None
        self.camera_controller = None  # To be set later
        self.debug_visualizer = None   # To be set later
        self.state_machine = None      # To be set later
        
        # Acquire Omniverse interfaces
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            
            # Subscribe to keyboard events
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
            )
            
            logger.info("✅ Keyboard input system initialized.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize keyboard input: {e}")
            self._keyboard = None
            self._keyboard_sub = None
    
    def set_camera_controller(self, camera_controller):
        """Sets the camera controller."""
        self.camera_controller = camera_controller
        logger.info("✅ Camera controller linked to keyboard handler.")
    
    def set_debug_visualizer(self, debug_visualizer):
        """Sets the debug visualization manager."""
        self.debug_visualizer = debug_visualizer
        logger.info("✅ Debug visualization manager linked to keyboard handler.")
    
    def set_state_machine(self, state_machine):
        """Sets the state machine."""
        self.state_machine = state_machine
        logger.info("✅ State machine linked to keyboard handler.")
    
    def set_scene_manager(self, scene_manager):
        """Sets the scene manager."""
        self.scene_manager = scene_manager
        logger.info("✅ Scene manager linked to keyboard handler.")

    def simulate_key_press(self, key):
        """
        Simulates a key press for automation scripts.
        
        Args:
            key (str): The key to simulate ('1', '2', '3', 'v', 'r', 'c', 'q').
        """
        key = key.lower()
        print(f"🤖 Simulating key press: {key}")

        if key.isdigit():
            if self.state_machine:
                self.state_machine.handle_key_input(key)
        elif key == 'v':
            if self.debug_visualizer:
                self.debug_visualizer.toggle_visibility()
            else:
                logger.warning("⚠️ Failed to simulate 'v' press: DebugVisualizer not set.")
        elif key == 'r':
            if self.state_machine:
                self.state_machine.reset_scene()
            elif hasattr(self, 'scene_manager') and self.scene_manager: # Fallback
                self.scene_manager.reset_scene()
            else:
                logger.warning("⚠️ Failed to simulate 'r' press: Neither SceneManager nor StateMachine is set.")
        elif key == 'c':
             if self.state_machine:
                self.state_machine.cancel_current_task()
        elif key == 'q':
            self.user_choice = 'q' # Used for external loop to detect exit
        else:
            logger.warning(f"⚠️ Unknown simulated key press: {key}")

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handles keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name
            print(f"🎹 Key pressed: {key_name}")
            
            # Handle number keys (KEY_1, KEY_2, KEY_3, NUMPAD_1, NUMPAD_2, NUMPAD_3)
            if key_name.startswith("KEY_") and key_name[4:].isdigit():
                digit = key_name[4:]
                self.user_choice = digit
                print(f"✅ Number key {digit} was recorded.")
                # If a state machine is present, pass the input directly to it
                if self.state_machine:
                    self.state_machine.handle_key_input(digit)
            elif key_name.startswith("NUMPAD_") and key_name[7:].isdigit():
                digit = key_name[7:]
                self.user_choice = digit
                print(f"✅ Number key {digit} was recorded.")
                # If a state machine is present, pass the input directly to it
                if self.state_machine:
                    self.state_machine.handle_key_input(digit)
            elif key_name.isdigit():
                self.user_choice = key_name
                print(f"✅ Number key {key_name} was recorded.")
                # If a state machine is present, pass the input directly to it
                if self.state_machine:
                    self.state_machine.handle_key_input(key_name)
                
            # Core function keys
            elif key_name == "R":
                self.user_choice = "R"
                print("✅ 'R' key was recorded (Reset Scene).")
                if self.state_machine:
                    self.state_machine.reset_scene()
                elif hasattr(self, 'scene_manager') and self.scene_manager:
                    self.scene_manager.reset_scene()

            elif key_name == "TAB":
                # TAB key: Switch camera view
                if self.camera_controller:
                    self.camera_controller.switch_camera()
                    print("✅ 'TAB' key: Switched camera view.")
                else:
                    print("⚠️ 'TAB' key: Camera controller not set.")
            elif key_name == "V":
                # V key: Toggle debug visualization
                if self.debug_visualizer:
                    self.debug_visualizer.toggle_visibility()
                    print("✅ 'V' key: Toggled debug visualization.")
                else:
                    print("⚠️ 'V' key: Debug visualizer not set.")
                    
            # Other function keys
            elif key_name == "F":
                self.user_choice = "F"
                print("✅ 'F' key was recorded.")
            elif key_name == "C":
                self.user_choice = "c"
                print("✅ 'C' key was recorded (Cancel Task).")
                if self.state_machine:
                    self.state_machine.cancel_current_task()
            elif key_name == "Q":
                self.user_choice = "q"
                print("✅ 'Q' key was recorded (Quit Program).")
            elif key_name == "Y":
                self.user_choice = "y"
                print("✅ 'Y' key was recorded.")
            elif key_name == "N":
                self.user_choice = "n"
                print("✅ 'N' key was recorded.")
                
        return True
    
    def get_user_choice(self):
        """Gets and consumes the user's input."""
        choice = self.user_choice
        self.user_choice = None
        return choice
    
    def peek_user_choice(self):
        """Peeks at the user's input without consuming it."""
        return self.user_choice
    
    def cleanup(self):
        """Cleans up the keyboard event subscription."""
        try:
            if hasattr(self, '_keyboard_sub') and self._keyboard_sub:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
                logger.info("✅ Keyboard event subscription cleaned up.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up keyboard event subscription: {e}")
