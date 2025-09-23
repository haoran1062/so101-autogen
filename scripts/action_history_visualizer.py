#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action History Visualization Script.
Reads action history data from a CSV file and plots the joint movement curves.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import argparse
import logging
from typing import Dict, List, Optional, Tuple
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActionHistoryVisualizer:
    """Visualizer for action history."""
    
    def __init__(self, csv_file_path: str):
        """Initializes the visualizer."""
        self.csv_file_path = csv_file_path
        self.data = None
        self.joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        
        # Figure related
        self.fig = None
        self.axes = None
        self.slider = None
        self.play_button = None
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Loads CSV data."""
        try:
            logger.info(f"📂 Loading CSV file: {self.csv_file_path}")
            
            if not os.path.exists(self.csv_file_path):
                raise FileNotFoundError(f"File not found: {self.csv_file_path}")
            
            # Read the CSV file
            self.data = pd.read_csv(self.csv_file_path)
            
            # Check for required columns
            required_columns = ['timestamp', 'step', 'round']
            for joint in self.joint_names:
                required_columns.extend([f'{joint}_position', f'{joint}_action'])
            
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"CSV file is missing required columns: {missing_columns}")
            
            # Get data information
            self.total_frames = len(self.data)
            self.current_frame = 0
            
            # Get round information
            self.rounds = sorted(self.data['round'].unique())
            self.current_round = self.rounds[0] if self.rounds else 1
            
            logger.info(f"📊 Data Information:")
            logger.info(f"   Total frames: {self.total_frames}")
            logger.info(f"   Number of rounds: {len(self.rounds)}")
            logger.info(f"   List of rounds: {self.rounds}")
            logger.info(f"   Data columns: {list(self.data.columns)}")
            
            # Display data statistics
            for joint in self.joint_names:
                pos_col = f'{joint}_position'
                action_col = f'{joint}_action'
                if pos_col in self.data.columns and action_col in self.data.columns:
                    pos_range = (self.data[pos_col].min(), self.data[pos_col].max())
                    action_range = (self.data[action_col].min(), self.data[action_col].max())
                    logger.info(f"   {joint}: Position range {pos_range}, Action range {action_range}")
            
            logger.info("✅ Data loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def create_visualization(self):
        """Creates the visualization interface."""
        try:
            logger.info("🎨 Creating visualization interface...")
            
            # Create figure and subplots
            self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
            self.fig.suptitle(f'Action History Visualization - {os.path.basename(self.csv_file_path)}', fontsize=16)
            
            # Set subplot titles
            for i, joint in enumerate(self.joint_names):
                row = i // 3
                col = i % 3
                self.axes[row, col].set_title(f'{joint.replace("_", " ").title()}')
            
            # Create control panel
            self.create_control_panel()
            
            # Display data
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
            
            # Create round selector
            if len(self.rounds) > 1:
                ax_round = plt.axes([0.1, 0.01, 0.3, 0.03])
                self.round_selector = RadioButtons(
                    ax_round, 
                    [f"Round {r}" for r in self.rounds],
                    active=0
                )
                self.round_selector.on_clicked(self.on_round_changed)
            
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
            
            # Plot charts for each joint
            for i, joint in enumerate(self.joint_names):
                row = i // 3
                col = i % 3
                ax = self.axes[row, col]
                
                pos_col = f'{joint}_position'
                action_col = f'{joint}_action'
                
                if pos_col in self.data.columns and action_col in self.data.columns:
                    # Get data for the current round
                    round_data = self.data[self.data['round'] == self.current_round]
                    
                    if len(round_data) > 0:
                        time_axis = np.arange(len(round_data))
                        
                        # Plot joint positions (dashed line)
                        ax.plot(time_axis, round_data[pos_col], 
                               color='blue', linestyle='--', 
                               label='Position', alpha=0.8, linewidth=2)
                        
                        # Plot action data (solid line)
                        ax.plot(time_axis, round_data[action_col], 
                               color='red', linestyle='-', 
                               label='Action', alpha=0.8, linewidth=2)
                        
                        # Mark current frame position
                        current_frame_in_round = min(frame_idx, len(round_data) - 1)
                        ax.axvline(x=current_frame_in_round, color='black', linestyle=':', alpha=0.8, linewidth=2)
                        
                        # Set labels and grid
                        ax.set_xlabel('Frame')
                        ax.set_ylabel('Position (radians)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # Set title
                        ax.set_title(f'{joint.replace("_", " ").title()}')
            
            # Update info display
            current_round_data = self.data[self.data['round'] == self.current_round]
            current_frame_in_round = min(frame_idx, len(current_round_data) - 1)
            info_text = f"Round: {self.current_round} | Frame: {current_frame_in_round + 1}/{len(current_round_data)} | Total Frames: {self.total_frames}"
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
    
    def on_round_changed(self, label):
        """Callback for round selection change."""
        round_num = int(label.split()[-1])
        if round_num != self.current_round:
            self.current_round = round_num
            self.current_frame = 0
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

def find_latest_csv_file(action_history_dir: str = "./action_history") -> Optional[str]:
    """Finds the latest CSV file."""
    try:
        if not os.path.exists(action_history_dir):
            logger.warning(f"Action history directory not found: {action_history_dir}")
            return None
        
        # Find all CSV files
        csv_pattern = os.path.join(action_history_dir, "action_history_*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            logger.warning(f"No action history CSV files found in {action_history_dir}")
            return None
        
        # Sort by modification time and return the latest
        latest_file = max(csv_files, key=os.path.getmtime)
        logger.info(f"Found the latest CSV file: {latest_file}")
        return latest_file
        
    except Exception as e:
        logger.error(f"❌ Failed to find CSV file: {e}")
        return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Action History Visualization Tool')
    parser.add_argument('--csv_file', help='Path to the CSV file (if not specified, the latest will be found automatically)')
    parser.add_argument('--action_history_dir', default='./action_history', help='Directory for action history files')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Determine the CSV file path
        csv_file_path = args.csv_file
        if csv_file_path is None:
            csv_file_path = find_latest_csv_file(args.action_history_dir)
            if csv_file_path is None:
                logger.error("❌ Could not find a CSV file to process.")
                sys.exit(1)
        
        # Create the visualizer
        visualizer = ActionHistoryVisualizer(csv_file_path)
        
        # Create the visualization interface
        visualizer.create_visualization()
        
        # Show the interface
        visualizer.show()
        
    except Exception as e:
        logger.error(f"❌ Program execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
