"""
Data Collection Manager - Optimized version based on previous work.
Resolves all known issues and provides reliable data collection functionality.
"""

import os
import logging
import numpy as np
import h5py
import torch
import time
import copy
import gc
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCollectionManager:
    """
    Data Collection Manager
    Based on the previously working version, with optimizations for performance and memory management,
    and fixes for all known issues.
    """
    
    def __init__(self, output_file_path: str, enable_data_collection: bool = True):
        """
        Initializes the Data Collection Manager.
        
        Args:
            output_file_path: The path for the output HDF5 file.
            enable_data_collection: Whether to enable data collection.
        """
        self.output_file_path = output_file_path
        self.enable_data_collection = enable_data_collection
        
        # HDF5 file related
        self.hdf5_file = None
        self.episode_count = 0
        
        # Current episode data
        self.current_episode_id = None
        self.current_episode_data = {
            'joint_pos': [],
            'actions': [],
            'front_images': [],
            'wrist_images': [],
            'timestamps': []
        }
        self.current_episode_start_time = None
        self.current_episode_stats = {}
        
        # User interaction related
        self.awaiting_confirmation = False
        self.confirmation_message = ""
        
        # Performance optimization related
        self._flush_counter = 0
        self._flush_interval = 50  # Flush every 50 frames
        
        # Initialize the HDF5 file
        if self.enable_data_collection:
            self._initialize_hdf5_file()
        
        logger.info(f"📊 Data Collection Manager initialized.")
        logger.info(f"   Output file: {output_file_path}")
        logger.info(f"   Data collection feature: {'Enabled' if enable_data_collection else 'Disabled'}")
    
    def _initialize_hdf5_file(self):
        """Initializes the HDF5 file and creates the basic structure."""
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
            
            # Check if the file already exists
            file_exists = os.path.exists(self.output_file_path)
            
            if file_exists:
                # Append mode
                self.hdf5_file = h5py.File(self.output_file_path, 'a')
                if 'data' not in self.hdf5_file:
                    self.hdf5_file.create_group('data')
                # Count existing episodes
                self.episode_count = len([key for key in self.hdf5_file['data'].keys() if key.startswith('demo_')])
                logger.info(f"📁 Existing file detected with {self.episode_count} episodes.")
            else:
                # Create a new file
                self.hdf5_file = h5py.File(self.output_file_path, 'w')
                self.hdf5_file.create_group('data')
                self.hdf5_file['data'].attrs['total'] = 0
                self.episode_count = 0
                logger.info(f"📁 Creating a new HDF5 file.")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize HDF5 file: {e}")
            self.enable_data_collection = False
    
    def start_episode(self, episode_id: str, target_info: Dict = None):
        """
        Starts a new episode.
        
        Args:
            episode_id: A unique identifier for the episode.
            target_info: Information about the target object (optional).
        """
        if not self.enable_data_collection:
            return
        
        self.current_episode_id = episode_id
        self.current_episode_start_time = datetime.now()
        
        # Reset current episode data
        self.current_episode_data = {
            'joint_pos': [],
            'actions': [],
            'front_images': [],
            'wrist_images': [],
            'timestamps': []
        }
        
        # Reset flush counter
        self._flush_counter = 0
        
        # Record episode statistics
        self.current_episode_stats = {
            'episode_id': episode_id,
            'target_info': target_info or {},
            'start_time': self.current_episode_start_time.isoformat(),
            'frame_count': 0
        }
        
        logger.info(f"🎬 Starting Episode: {episode_id}")
        if target_info:
            logger.info(f"   Target: {target_info.get('name', 'unknown')}")
    
    def record_frame(self, 
                    joint_positions: np.ndarray,
                    actions: np.ndarray,
                    front_image: Optional[np.ndarray] = None,
                    wrist_image: Optional[np.ndarray] = None,
                    timestamp: Optional[float] = None):
        """
        Records a single frame of data.
        
        Args:
            joint_positions: Joint positions (6D, radians).
            actions: Action commands (6D, radians).
            front_image: Front camera image (480, 640, 3, uint8).
            wrist_image: Wrist camera image (480, 640, 3, uint8).
            timestamp: Timestamp (optional).
        """
        if not self.enable_data_collection or self.current_episode_id is None:
            return
        
        
        try:
            # Validate data format
            if joint_positions.shape != (6,):
                logger.warning(f"⚠️ Incorrect joint position dimensions: {joint_positions.shape}, expected (6,)")
                return
            
            if actions.shape != (6,):
                logger.warning(f"⚠️ Incorrect action command dimensions: {actions.shape}, expected (6,)")
                return
            
            # Record data
            self.current_episode_data['joint_pos'].append(joint_positions.astype(np.float32))
            self.current_episode_data['actions'].append(actions.astype(np.float32))
            
            # Record images (if provided)
            if front_image is not None:
                # Handle RGBA to RGB conversion
                if front_image.shape == (480, 640, 4):
                    # RGBA format, extract RGB channels
                    front_image = front_image[:, :, :3]
                elif front_image.shape != (480, 640, 3):
                    logger.warning(f"⚠️ Incorrect front image dimensions: {front_image.shape}, expected (480, 640, 3) or (480, 640, 4)")
                
                if front_image.shape == (480, 640, 3):
                    self.current_episode_data['front_images'].append(front_image.astype(np.uint8))
            
            if wrist_image is not None:
                # Handle RGBA to RGB conversion
                if wrist_image.shape == (480, 640, 4):
                    # RGBA format, extract RGB channels
                    wrist_image = wrist_image[:, :, :3]
                elif wrist_image.shape != (480, 640, 3):
                    logger.warning(f"⚠️ Incorrect wrist image dimensions: {wrist_image.shape}, expected (480, 640, 3) or (480, 640, 4)")
                
                if wrist_image.shape == (480, 640, 3):
                    self.current_episode_data['wrist_images'].append(wrist_image.astype(np.uint8))
            
            # Record timestamp
            if timestamp is None:
                timestamp = len(self.current_episode_data['timestamps']) / 30.0  # Assume 30fps
            self.current_episode_data['timestamps'].append(timestamp)
            
            # Update statistics
            self.current_episode_stats['frame_count'] += 1
            self._flush_counter += 1
            
            # Periodically flush the file (performance optimization)
            if self._flush_counter % self._flush_interval == 0:
                self.hdf5_file.flush()
                logger.debug(f"📊 Data recording progress: frame_count={self.current_episode_stats['frame_count']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record frame data: {e}")
    
    def end_episode(self, success: bool, final_stats: Dict = None):
        """
        Ends the current episode and prepares for user confirmation.
        
        Args:
            success: Whether the episode was successful.
            final_stats: Final statistics.
        """
        if not self.enable_data_collection or self.current_episode_id is None:
            return
        
        # Calculate episode statistics
        episode_duration = datetime.now() - self.current_episode_start_time
        frame_count = len(self.current_episode_data['joint_pos'])
        
        # Update statistics
        self.current_episode_stats.update({
            'success': success,
            'duration_seconds': episode_duration.total_seconds(),
            'frame_count': frame_count,
            'end_time': datetime.now().isoformat()
        })
        
        if final_stats:
            self.current_episode_stats.update(final_stats)
        
        # Prepare confirmation message
        self.confirmation_message = self._format_confirmation_message()
        self.awaiting_confirmation = True
        
        logger.info(f"🎬 Episode ended: {self.current_episode_id}")
        logger.info(f"   Success: {success}, Duration: {episode_duration.total_seconds():.1f}s, Frames: {frame_count}")
    
    def _format_confirmation_message(self) -> str:
        """Formats the confirmation message."""
        stats = self.current_episode_stats
        
        message = f"""
=== Episode {stats['episode_id']} Grasp Completed ===
Target: {stats.get('target_info', {}).get('name', 'unknown')}
Grasp Position: {stats.get('target_info', {}).get('position', 'unknown')}
Grasp Time: {stats.get('duration_seconds', 0):.1f}s
Frames: {stats.get('frame_count', 0)}
Success Rate: {'100%' if stats.get('success', False) else '0%'}

Save this data? [Y/N/Q]
"""
        return message
    
    def handle_user_confirmation(self, user_choice: str) -> str:
        """
        Handles the user's confirmation choice.
        
        Args:
            user_choice: The user's choice ('y', 'n', 'q').
        
        Returns:
            The result of the action ('save', 'discard', 'quit', 'waiting').
        """
        if not self.awaiting_confirmation:
            return 'waiting'
        
        user_choice = user_choice.lower()
        
        if user_choice == 'y':
            return self.save_episode()
        elif user_choice == 'n':
            return self.discard_episode()
        elif user_choice == 'q':
            return self.quit_collection()
        else:
            return 'waiting'
    
    def save_episode(self) -> str:
        """Saves the current episode data."""
        if not self.enable_data_collection or self.current_episode_id is None:
            return 'error'
        
        try:
            # Check if the episode was successful, only save successful ones
            success = self.current_episode_stats.get('success', False)
            if not success:
                logger.info("❌ Episode was not successful, not saving data.")
                self._clear_episode_cache()
                return "failed_episode"
            
            print("💾 Starting to save episode data...")
            logger.info("💾 Starting to save episode data...")
            
            
            # Create episode group
            demo_group_name = f"demo_{self.episode_count}"
            demo_group = self.hdf5_file['data'].create_group(demo_group_name)
            
            # Create obs group
            obs_group = demo_group.create_group('obs')
            
            # Save joint data
            if self.current_episode_data['joint_pos']:
                joint_data = np.array(self.current_episode_data['joint_pos'], dtype=np.float32)
                obs_group.create_dataset('joint_pos', data=joint_data)
            
            # Save action data
            if self.current_episode_data['actions']:
                actions_data = np.array(self.current_episode_data['actions'], dtype=np.float32)
                obs_group.create_dataset('actions', data=actions_data)
            
            # Save image data
            if self.current_episode_data['front_images']:
                front_data = np.array(self.current_episode_data['front_images'], dtype=np.uint8)
                obs_group.create_dataset('front', data=front_data)
            
            if self.current_episode_data['wrist_images']:
                wrist_data = np.array(self.current_episode_data['wrist_images'], dtype=np.uint8)
                obs_group.create_dataset('wrist', data=wrist_data)
            
            # Save episode attributes
            for key, value in self.current_episode_stats.items():
                if isinstance(value, (str, int, float, bool)):
                    demo_group.attrs[key] = value
            
            # Set success attribute (required by leisaac format)
            demo_group.attrs['success'] = self.current_episode_stats.get('success', False)
            
            # Flush the file
            self.hdf5_file.flush()
            
            self.episode_count += 1
            
            print("✅ Episode data saved successfully!")
            logger.info(f"✅ Episode data saved: {demo_group_name}")
            
            # Reset current episode
            self._reset_current_episode()
            
            return 'save'
            
        except Exception as e:
            logger.error(f"❌ Failed to save episode: {e}")
            return 'error'
    
    def discard_episode(self) -> str:
        """Discards the current episode data."""
        logger.info(f"🗑️ Episode data discarded: {self.current_episode_id}")
        self._reset_current_episode()
        return 'discard'
    
    def quit_collection(self) -> str:
        """Quits data collection."""
        logger.info("🛑 User chose to quit data collection.")
        self.close()
        return 'quit'
    
    def _reset_current_episode(self):
        """Resets the state of the current episode."""
        self.current_episode_id = None
        self.current_episode_data = {
            'joint_pos': [],
            'actions': [],
            'front_images': [],
            'wrist_images': [],
            'timestamps': []
        }
        self.current_episode_stats = {}
        self.awaiting_confirmation = False
        self.confirmation_message = ""
        self._flush_counter = 0
    
    def _clear_episode_cache(self):
        """Clears the episode cache to free up memory."""
        try:
            # Clear current episode data
            self.current_episode_data = {}
            self.current_episode_stats = {}
            
            # Force garbage collection
            gc.collect()
            
            logger.debug("🧹 Episode cache cleared.")
        except Exception as e:
            logger.warning(f"⚠️ Warning occurred while clearing cache: {e}")
    
    def get_confirmation_message(self) -> str:
        """Gets the current confirmation message."""
        return self.confirmation_message if self.awaiting_confirmation else ""
    
    def is_awaiting_confirmation(self) -> bool:
        """Checks if the manager is awaiting user confirmation."""
        return self.awaiting_confirmation
    
    def get_episode_count(self) -> int:
        """Gets the number of saved episodes."""
        return self.episode_count
    
    def close(self):
        """Closes the HDF5 file."""
        try:
            if self.hdf5_file is not None:
                try:
                    self.hdf5_file.close()
                    logger.info(f"📁 HDF5 file closed: {self.output_file_path}")
                    logger.info(f"   Total of {self.episode_count} episodes saved.")
                except Exception as e:
                    logger.error(f"❌ Failed to close HDF5 file: {e}")
                finally:
                    self.hdf5_file = None
        except Exception as e:
            logger.error(f"❌ Failed to close Data Collection Manager: {e}")
        finally:
            # Ensure all references are cleared
            self.enable_data_collection = False
    
    def __del__(self):
        """Destructor to ensure the file is closed."""
        self.close()
