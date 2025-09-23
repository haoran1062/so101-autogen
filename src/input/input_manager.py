# -*- coding: utf-8 -*-
"""
Input Manager
Coordinates and manages keyboard inputs and related functionalities.
"""

import logging

logger = logging.getLogger(__name__)

class InputManager:
    """Unified Input Manager."""
    
    def __init__(self):
        """Initializes the Input Manager."""
        self.keyboard_handler = None
        self.camera_controller = None
        self.scene_manager = None  # Scene manager (for 'R' key reset)
        self.ik_controller = None  # IK controller (for arm reset)
        
        logger.info("✅ Input Manager initialized.")
    
    def setup(self, keyboard_handler=None, camera_controller=None, 
              scene_manager=None, ik_controller=None):
        """
        Sets up the various controllers.
        
        Args:
            keyboard_handler: The keyboard handler.
            camera_controller: The camera controller.
            scene_manager: The scene manager.
            ik_controller: The IK controller.
        """
        if keyboard_handler:
            self.keyboard_handler = keyboard_handler
            
        if camera_controller:
            self.camera_controller = camera_controller
            # Associate the camera controller with the keyboard handler.
            if self.keyboard_handler:
                self.keyboard_handler.set_camera_controller(camera_controller)
                
        if scene_manager:
            self.scene_manager = scene_manager
            
        if ik_controller:
            self.ik_controller = ik_controller
            
        logger.info("✅ Input Manager setup complete.")
        logger.info(f"   Keyboard Handler: {'Set' if self.keyboard_handler else 'Not set'}")
        logger.info(f"   Camera Controller: {'Set' if self.camera_controller else 'Not set'}")
        logger.info(f"   Scene Manager: {'Set' if self.scene_manager else 'Not set'}")
        logger.info(f"   IK Controller: {'Set' if self.ik_controller else 'Not set'}")
    
    def process_input(self):
        """Processes user input (called every frame)."""
        if not self.keyboard_handler:
            return
        
        # Get user input.
        choice = self.keyboard_handler.get_user_choice()
        # Note: 'v', 'tab', and number keys are now handled directly by KeyboardHandler.
        # InputManager only handles commands that change the main flow (R, Q, C).
        if choice:
            self._handle_user_choice(choice)
    
    def _handle_user_choice(self, choice):
        """Handles the user's choice."""
        if choice == "R":
            # 'R' key: Reset the scene.
            self._handle_scene_reset()
        elif choice == "Q" or choice == "q":
            # 'Q' key: Quit the program.
            self._handle_quit()
        elif choice == "C" or choice == "c":
            # 'C' key: Cancel the current operation.
            self._handle_cancel()
        else:
            # Other keys: Log but do not process.
            logger.info(f"🎹 Key pressed: {choice}")
    
    def _handle_scene_reset(self):
        """Handles scene reset ('R' key)."""
        print("🔄 'R' key: Initiating scene reset...")
        logger.info("🔄 'R' key: Initiating scene reset.")
        
        try:
            # 1. Reset object positions (if scene manager is available).
            if self.scene_manager and hasattr(self.scene_manager, 'reset_scene'):
                self.scene_manager.reset_scene()
                print("✅ Object positions have been reset.")
            else:
                print("⚠️ Scene manager not set, skipping object reset.")
            
            # 2. Return arm to initial pose (if IK controller is available).
            if self.ik_controller and hasattr(self.ik_controller, 'move_to_initial_position'):
                self.ik_controller.move_to_initial_position()
                print("✅ Robot arm has returned to its initial pose.")
            else:
                print("⚠️ IK controller not set, skipping arm reset.")
                
            print("🎉 Scene reset complete!")
            
        except Exception as e:
            logger.error(f"❌ Scene reset failed: {e}")
            print(f"❌ Scene reset failed: {e}")
    
    def _handle_quit(self):
        """Handles program exit ('Q' key)."""
        print("👋 'Q' key: Preparing to exit...")
        logger.info("👋 'Q' key: Preparing to exit.")
        # Exit logic, like saving data, can be added here.
    
    def _handle_cancel(self):
        """Handles operation cancellation ('C' key)."""
        print("❌ 'C' key: Canceling current operation.")
        logger.info("❌ 'C' key: Canceling current operation.")
        # Cancellation logic can be added here.
    
    def cleanup(self):
        """Cleans up the Input Manager."""
        if self.keyboard_handler and hasattr(self.keyboard_handler, 'cleanup'):
            self.keyboard_handler.cleanup()
        logger.info("✅ Input Manager cleaned up.")
