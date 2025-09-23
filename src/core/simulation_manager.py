# -*- coding: utf-8 -*-
"""
Simulation Manager
Handles the lifecycle of the Isaac Sim application, including startup,
extension management, and cleanup.
"""

import os
import sys
import signal
import logging
from typing import Optional

# This logic was retained from the project's early development for Windows compatibility.
def setup_windows_encoding():
    """Sets up the console encoding for Windows systems to better support UTF-8."""
    if os.name == 'nt':  # Windows system
        try:
            # Set console code page to UTF-8
            os.system('chcp 65001 > nul')
            
            # Set environment variables
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
            
            # Try to set the locale
            import locale
            try:
                locale.setlocale(locale.LC_ALL, 'C.UTF-8')
            except:
                try:
                    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
                except:
                    pass
            
            # Reconfigure stdout and stderr
            if hasattr(sys.stdout, 'reconfigure'):
                try:
                    sys.stdout.reconfigure(encoding='utf-8')
                except:
                    pass
            if hasattr(sys.stderr, 'reconfigure'):
                try:
                    sys.stderr.reconfigure(encoding='utf-8')
                except:
                    pass
                    
        except Exception as e:
            pass  # Ignore encoding setup errors

# This function was retained from the project's early development for Windows compatibility.
def safe_print(*args, **kwargs):
    """A safe print function to avoid Unicode encoding errors, especially on Windows."""
    try:
        # Try to print normally
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # If it fails, retry after cleaning Unicode characters
        try:
            import re
            cleaned_args = []
            for arg in args:
                if isinstance(arg, str):
                    # Remove all Unicode emojis and special characters
                    cleaned_arg = re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', arg)
                    cleaned_args.append(cleaned_arg)
                else:
                    cleaned_args.append(arg)
            print(*cleaned_args, **kwargs)
        except Exception:
            # If it still fails, use the simplest output
            print("PRINT:", *args)

class SimulationManager:
    """
    Simulation Manager
    Responsible for starting the Isaac Sim application, managing extensions, and cleanup.
    """
    
    def __init__(self, headless: bool = False):
        """
        Initializes the SimulationManager.
        
        Args:
            headless: Whether to run in headless mode.
        """
        self.headless = headless
        self.simulation_app = None
        self.ext_manager = None
        
        # Set a global variable for the signal handler to access.
        global global_simulation_app
        global_simulation_app = None
        
        # Set up console encoding for Windows.
        setup_windows_encoding()
        
        # Disable Unicode emojis in Isaac Sim console output for cleaner logs.
        os.environ['DISABLE_UNICODE_EMOJIS'] = '1'
        
        safe_print("SimulationManager initialized.")
    
    def start_simulation(self):
        """
        Starts the Isaac Sim simulation application.
        """
        try:
            from isaacsim import SimulationApp
            
            self.simulation_app = SimulationApp({"headless": self.headless})
            
            # Set the global reference for the signal handler.
            global global_simulation_app
            global_simulation_app = self.simulation_app
            
            safe_print(f"✅ Isaac Sim application started (headless: {self.headless})")
            
            # Load necessary Isaac Sim extensions.
            self._load_extensions()
            
            return self.simulation_app
            
        except Exception as e:
            safe_print(f"❌ Failed to start Isaac Sim application: {e}")
            raise
    
    def _load_extensions(self):
        """
        Loads necessary Isaac Sim extensions.
        """
        try:
            import omni.kit.app
            
            self.ext_manager = omni.kit.app.get_app().get_extension_manager()
            self.ext_manager.set_extension_enabled_immediate("omni.isaac.motion_generation", True)
            self.ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
            
            safe_print("✅ Isaac Sim extensions loaded successfully.")
            
        except Exception as e:
            safe_print(f"❌ Failed to load Isaac Sim extensions: {e}")
            raise
    
    def is_running(self) -> bool:
        """Checks if the simulation is running."""
        if self.simulation_app is None:
            return False
        return self.simulation_app.is_running()
    
    def close(self):
        """
        Closes the simulation application.
        """
        if self.simulation_app is not None:
            try:
                self.simulation_app.close()
                safe_print("✅ Isaac Sim application closed successfully.")
            except Exception as e:
                safe_print(f"⚠️ Warning occurred while closing simulation application: {e}")
            finally:
                self.simulation_app = None
                
                # Clear the global reference
                global global_simulation_app
                global_simulation_app = None

# Global variable for the signal handler.
global_simulation_app = None

def signal_handler(signum, frame):
    """
    Handles signals like Ctrl+C to ensure the program exits safely.
    """
    global global_simulation_app
    
    safe_print(f"\nReceived signal {signum}, exiting safely...")
    
    # Close the simulation application
    if global_simulation_app is not None:
        try:
            global_simulation_app.close()
            print("✅ Simulation application closed safely.")
        except Exception as e:
            print(f"⚠️ Warning occurred while closing simulation application: {e}")
    
    safe_print("Program exited safely.")
    sys.exit(0)

# Register signal handlers to ensure clean shutdown.
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
