# -*- coding: utf-8 -*-
"""
Extension Loading Utility Module
Provides functionality for loading Isaac Sim extensions, extracted from the main script.
"""


class ExtensionLoader:
    """Isaac Sim Extension Loader"""
    
    @staticmethod
    def load_required_extensions():
        """Loads the required Isaac Sim extensions.
        
        Loads extensions in the standard way, as done in the main script.
        """
        print("🔧 Loading Isaac Sim extensions...")
        
        try:
            # Import the extension manager
            import omni.kit.app
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            
            # Load the motion generation extension
            ext_manager.set_extension_enabled_immediate("omni.isaac.motion_generation", True)
            print("✅ omni.isaac.motion_generation extension has been loaded.")
            
            # Load the core extension
            ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
            print("✅ omni.isaac.core extension has been loaded.")
            
            print("✅ All required extensions have been loaded successfully.")
            
        except Exception as e:
            print(f"❌ Failed to load extensions: {e}")
            raise
    
    @staticmethod
    def import_required_modules():
        """Imports modules that are required after extensions have been loaded.
        
        These modules must be imported after the extensions are loaded.
        """
        print("📦 Importing extension-related modules...")
        
        try:
            # Import motion generation related modules
            from omni.isaac.motion_generation import LulaKinematicsSolver
            print("✅ LulaKinematicsSolver has been imported.")
            
            # Import bounds computation related modules
            from omni.isaac.core.utils.bounds import create_bbox_cache, compute_obb, compute_combined_aabb
            print("✅ Bounds computation modules have been imported.")
            
            # Import debug draw module
            from isaacsim.util.debug_draw import _debug_draw
            print("✅ Debug draw module has been imported.")
            
            # Import camera related modules
            from omni.kit.viewport.utility import get_viewport_from_window_name
            from isaacsim.sensors.camera import Camera
            print("✅ Camera modules have been imported.")
            
            # Import carb module
            import carb
            print("✅ carb module has been imported.")
            
            print("✅ All extension-related modules have been imported successfully.")
            
            # Return the imported modules for use by other modules
            return {
                'LulaKinematicsSolver': LulaKinematicsSolver,
                'create_bbox_cache': create_bbox_cache,
                'compute_obb': compute_obb,
                'compute_combined_aabb': compute_combined_aabb,
                '_debug_draw': _debug_draw,
                'get_viewport_from_window_name': get_viewport_from_window_name,
                'Camera': Camera,
                'carb': carb
            }
            
        except Exception as e:
            print(f"❌ Failed to import modules: {e}")
            raise
    
    @staticmethod
    def load_all():
        """Loads all extensions and modules.
        
        This is the main entry point, loading everything in the correct order.
        """
        print("🚀 Starting to load Isaac Sim extensions and modules...")
        
        # 1. Load extensions
        ExtensionLoader.load_required_extensions()
        
        # 2. Import modules
        modules = ExtensionLoader.import_required_modules()
        
        print("✅ Isaac Sim extensions and modules have been loaded successfully.")
        return modules


# Compatibility function to maintain the same interface as the main script.
def load_required_extensions():
    """Compatibility function."""
    return ExtensionLoader.load_all()
