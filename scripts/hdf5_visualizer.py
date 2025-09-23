#!/usr/bin/env python3
"""
HDF5 Data Visualization Script.
Supports viewing data from multiple episodes, timeline control, and a 2x2 display layout.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import argparse
import logging
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HDF5Visualizer:
    """Visualizer for HDF5 data."""
    
    def __init__(self, hdf5_file_path: str):
        """Initializes the visualizer."""
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = None
        self.data = {}
        self.episodes = []
        self.current_episode_idx = 0
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.play_speed = 1.0  # Playback speed multiplier
        
        # Figure related
        self.fig = None
        self.axes = None
        self.slider = None
        self.play_button = None
        self.episode_selector = None
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Loads HDF5 data."""
        try:
            logger.info(f"📂 Loading HDF5 file: {self.hdf5_file_path}")
            
            if not os.path.exists(self.hdf5_file_path):
                raise FileNotFoundError(f"File not found: {self.hdf5_file_path}")
            
            self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
            
            # Check file structure
            if 'data' not in self.hdf5_file:
                raise ValueError("HDF5 file is missing the 'data' group")
            
            # Get all episodes
            self.episodes = list(self.hdf5_file['data'].keys())
            logger.info(f"📊 Found {len(self.episodes)} episodes: {self.episodes}")
            
            if not self.episodes:
                raise ValueError("No episode data found")
            
            # Load data for the first episode
            self.load_episode_data(0)
            
            logger.info("✅ Data loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def load_episode_data(self, episode_idx: int):
        """Loads data for a specific episode."""
        try:
            episode_name = self.episodes[episode_idx]
            episode_group = self.hdf5_file['data'][episode_name]
            
            logger.info(f"📊 Loading episode: {episode_name}")
            
            # Check episode structure
            if 'obs' not in episode_group:
                raise ValueError(f"Episode {episode_name} is missing the 'obs' group")
            
            obs_group = episode_group['obs']
            
            # Load data
            self.data = {
                'joint_pos': obs_group['joint_pos'][:] if 'joint_pos' in obs_group else None,
                'actions': obs_group['actions'][:] if 'actions' in obs_group else None,
                'front': obs_group['front'][:] if 'front' in obs_group else None,
                'wrist': obs_group['wrist'][:] if 'wrist' in obs_group else None,
            }
            
            # Get data info
            self.total_frames = len(self.data['joint_pos']) if self.data['joint_pos'] is not None else 0
            self.current_frame = 0
            
            # Get episode attributes
            self.episode_success = episode_group.attrs.get('success', False)
            self.episode_seed = episode_group.attrs.get('seed', None)
            
            logger.info(f"📊 Episode Information:")
            logger.info(f"   Frames: {self.total_frames}")
            logger.info(f"   Success: {self.episode_success}")
            logger.info(f"   Seed: {self.episode_seed}")
            
            if self.data['joint_pos'] is not None:
                logger.info(f"   Joint data shape: {self.data['joint_pos'].shape}")
            if self.data['actions'] is not None:
                logger.info(f"   Action data shape: {self.data['actions'].shape}")
            if self.data['front'] is not None:
                logger.info(f"   Front image shape: {self.data['front'].shape}")
            if self.data['wrist'] is not None:
                logger.info(f"   Wrist image shape: {self.data['wrist'].shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load episode data: {e}")
            raise
    
    def create_visualization(self):
        """Creates the visualization interface."""
        try:
            logger.info("🎨 Creating visualization interface...")
            
            # Create figure and subplots
            self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
            self.fig.suptitle(f'HDF5 Data Visualization - {os.path.basename(self.hdf5_file_path)}', fontsize=16)
            
            # Set subplot titles
            self.axes[0, 0].set_title('Front Camera')
            self.axes[0, 1].set_title('Wrist Camera')
            self.axes[1, 0].set_title('Base Joints (shoulder_pan, shoulder_lift, elbow_flex)')
            self.axes[1, 1].set_title('End Joints (wrist_flex, wrist_roll, gripper)')
            
            # Create control panel
            self.create_control_panel()
            
            # Display the first frame
            self.update_display()
            
            logger.info("✅ Visualization interface created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create visualization interface: {e}")
            raise
    
    def create_control_panel(self):
        """Creates the control panel."""
        try:
            # Adjust layout to make space for the control panel
            plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, top=0.9)
            
            # Create timeline slider
            ax_slider = plt.axes([0.1, 0.05, 0.6, 0.03])
            self.slider = Slider(
                ax_slider, 'Frame', 0, max(0, self.total_frames - 1),
                valinit=0, valstep=1
            )
            self.slider.on_changed(self.on_slider_changed)
            
            # Create play button
            ax_play = plt.axes([0.75, 0.05, 0.08, 0.03])
            self.play_button = Button(ax_play, 'Play')
            self.play_button.on_clicked(self.on_play_clicked)
            
            # Create step buttons
            ax_prev = plt.axes([0.85, 0.05, 0.05, 0.03])
            ax_next = plt.axes([0.92, 0.05, 0.05, 0.03])
            self.prev_button = Button(ax_prev, '◀')
            self.next_button = Button(ax_next, '▶')
            self.prev_button.on_clicked(self.on_prev_clicked)
            self.next_button.on_clicked(self.on_next_clicked)
            
            # Create episode selector
            if len(self.episodes) > 1:
                ax_episode = plt.axes([0.1, 0.01, 0.3, 0.03])
                self.episode_selector = RadioButtons(
                    ax_episode, 
                    [f"Episode {i+1}: {ep}" for i, ep in enumerate(self.episodes)],
                    active=0
                )
                self.episode_selector.on_clicked(self.on_episode_changed)
            
            # Create info display
            self.info_text = self.fig.text(0.5, 0.02, '', ha='center', fontsize=10)
            
        except Exception as e:
            logger.error(f"❌ Failed to create control panel: {e}")
            raise
    
    def update_display(self):
        """Updates the display content."""
        try:
            if self.total_frames == 0:
                return
            
            frame_idx = min(self.current_frame, self.total_frames - 1)
            
            # Clear all subplots
            for ax in self.axes.flat:
                ax.clear()
            
            # Display front camera image
            if self.data['front'] is not None and frame_idx < len(self.data['front']):
                self.axes[0, 0].imshow(self.data['front'][frame_idx])
                self.axes[0, 0].set_title(f'Front Camera (Frame {frame_idx + 1})')
                self.axes[0, 0].axis('off')
            
            # Display wrist camera image
            if self.data['wrist'] is not None and frame_idx < len(self.data['wrist']):
                self.axes[0, 1].imshow(self.data['wrist'][frame_idx])
                self.axes[0, 1].set_title(f'Wrist Camera (Frame {frame_idx + 1})')
                self.axes[0, 1].axis('off')
            
            # Display base joint data (bottom left)
            if self.data['joint_pos'] is not None and self.data['actions'] is not None:
                joint_data = self.data['joint_pos']
                action_data = self.data['actions']
                time_axis = np.arange(len(joint_data))
                
                # Define joint names and colors
                base_joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']
                colors = ['blue', 'red', 'green']
                
                for i in range(3):  # Only show the first 3 joints (base joints)
                    # Plot joint positions (dashed line)
                    self.axes[1, 0].plot(time_axis, joint_data[:, i], 
                                        color=colors[i], linestyle='--', 
                                        label=f'{base_joint_names[i]} (joint)', alpha=0.8)
                    
                    # Plot action data (solid line)
                    self.axes[1, 0].plot(time_axis, action_data[:, i], 
                                        color=colors[i], linestyle='-', 
                                        label=f'{base_joint_names[i]} (action)', alpha=0.8)
                
                # Mark current frame position
                self.axes[1, 0].axvline(x=frame_idx, color='black', linestyle=':', alpha=0.8)
                self.axes[1, 0].set_xlabel('Frame')
                self.axes[1, 0].set_ylabel('Position (radians)')
                self.axes[1, 0].legend()
                self.axes[1, 0].grid(True, alpha=0.3)
            
            # Display end effector joint data (bottom right)
            if self.data['joint_pos'] is not None and self.data['actions'] is not None:
                joint_data = self.data['joint_pos']
                action_data = self.data['actions']
                time_axis = np.arange(len(joint_data))
                
                # Define joint names and colors
                end_joint_names = ['wrist_flex', 'wrist_roll', 'gripper']
                colors = ['purple', 'orange', 'brown']
                
                for i in range(3):  # Show the last 3 joints (end effector joints)
                    joint_idx = i + 3  # Start from the 4th joint
                    # Plot joint positions (dashed line)
                    self.axes[1, 1].plot(time_axis, joint_data[:, joint_idx], 
                                        color=colors[i], linestyle='--', 
                                        label=f'{end_joint_names[i]} (joint)', alpha=0.8)
                    
                    # Plot action data (solid line)
                    self.axes[1, 1].plot(time_axis, action_data[:, joint_idx], 
                                        color=colors[i], linestyle='-', 
                                        label=f'{end_joint_names[i]} (action)', alpha=0.8)
                
                # Mark current frame position
                self.axes[1, 1].axvline(x=frame_idx, color='black', linestyle=':', alpha=0.8)
                self.axes[1, 1].set_xlabel('Frame')
                self.axes[1, 1].set_ylabel('Position (radians)')
                self.axes[1, 1].legend()
                self.axes[1, 1].grid(True, alpha=0.3)
            
            # Update info display
            episode_name = self.episodes[self.current_episode_idx]
            info_text = f"Episode: {episode_name} | Frame: {frame_idx + 1}/{self.total_frames} | Success: {self.episode_success}"
            if self.episode_seed is not None:
                info_text += f" | Seed: {self.episode_seed}"
            self.info_text.set_text(info_text)
            
            # Update slider position (avoiding recursive calls)
            if self.slider.val != frame_idx:
                self.slider.set_val(frame_idx)
            
            # Redraw the figure
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            logger.error(f"❌ Failed to update display: {e}")
    
    def on_slider_changed(self, val):
        """Callback for slider value change."""
        self.current_frame = int(val)
        self.update_display()
    
    def on_play_clicked(self, event):
        """Callback for play button click."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.label.set_text('Pause')
            self.start_animation()
        else:
            self.play_button.label.set_text('Play')
            self.stop_animation()
    
    def on_prev_clicked(self, event):
        """Callback for previous frame button click."""
        self.current_frame = max(0, self.current_frame - 1)
        self.update_display()
    
    def on_next_clicked(self, event):
        """Callback for next frame button click."""
        self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        self.update_display()
    
    def on_episode_changed(self, label):
        """Callback for episode selection change."""
        episode_idx = int(label.split(':')[0].split()[-1]) - 1
        if episode_idx != self.current_episode_idx:
            self.current_episode_idx = episode_idx
            self.load_episode_data(episode_idx)
            self.update_display()
    
    def start_animation(self):
        """Starts the animation playback."""
        def animate(frame):
            if self.is_playing:
                self.current_frame = (self.current_frame + 1) % self.total_frames
                self.update_display()
        
        self.anim = animation.FuncAnimation(
            self.fig, animate, frames=None, 
            interval=100, repeat=False
        )
    
    def stop_animation(self):
        """Stops the animation playback."""
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()
    
    def show(self):
        """Shows the visualization interface."""
        try:
            logger.info("🎬 Showing visualization interface...")
            plt.show()
        except Exception as e:
            logger.error(f"❌ Failed to show interface: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleans up resources."""
        try:
            if self.hdf5_file is not None:
                self.hdf5_file.close()
            logger.info("✅ Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"❌ Failed to clean up resources: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='HDF5 Data Visualization Tool')
    parser.add_argument('--hdf5_file', help='Path to the HDF5 file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create the visualizer
        visualizer = HDF5Visualizer(args.hdf5_file)
        
        # Create the visualization interface
        visualizer.create_visualization()
        
        # Show the interface
        visualizer.show()
        
    except Exception as e:
        logger.error(f"❌ Program execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
