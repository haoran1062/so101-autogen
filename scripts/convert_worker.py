import os
import sys
import numpy as np
import argparse
import tempfile
import shutil
import h5py
from tqdm import tqdm
import json

"""
This script is a worker process designed for data conversion. It is intended to
be called by an orchestrator script (like `parallel_converter.py`) and is not
meant for direct user execution for parallel processing. It handles the conversion
of a single demo from an HDF5 file.
"""

# Feature definition for the single-arm so101_follower configuration
SINGLE_ARM_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

    },
    "observation.images.front": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    }
}

# Feature definition for the bi-arm so101_follower configuration
BI_ARM_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ]
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ]

    },
    "observation.images.left": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.top": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.right": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    }
}

# Define joint position limits for preprocessing
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10, 100.0),
]
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (0, 100),
]


def preprocess_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Remaps joint positions from Isaac Sim's range to LeRobot's expected range."""
    joint_pos = joint_pos / np.pi * 180
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        joint_pos[:, i] = (joint_pos[:, i] - isaaclab_min) / (isaaclab_max - isaaclab_min) * (lerobot_max - lerobot_min) + lerobot_min
    return joint_pos


def process_single_arm_data(dataset: 'LeRobotDataset', task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    """Processes and adds data for a single-arm robot to the dataset."""
    try:
        actions = np.array(demo_group['obs/actions'])
        joint_pos = np.array(demo_group['obs/joint_pos'])
        front_images = np.array(demo_group['obs/front'])
        wrist_images = np.array(demo_group['obs/wrist'])
    except KeyError:
        print(f'Demo {demo_name} is not valid and will be skipped.', file=sys.stderr)
        return False

    # Preprocess actions and joint positions
    actions = preprocess_joint_pos(actions)
    joint_pos = preprocess_joint_pos(joint_pos)

    assert actions.shape[0] == joint_pos.shape[0] == front_images.shape[0] == wrist_images.shape[0]
    total_state_frames = actions.shape[0]
    # Skip the first 5 frames to avoid initial artifacts
    for frame_index in range(5, total_state_frames):
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
            "observation.images.front": front_images[frame_index],
            "observation.images.wrist": wrist_images[frame_index],
        }
        # Let the conductor handle episode_index and global index during post-processing
        dataset.add_frame(frame=frame, task=task)

    return True


def process_bi_arm_data(dataset: 'LeRobotDataset', task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    """Processes and adds data for a dual-arm robot to the dataset."""
    try:
        actions = np.array(demo_group['obs/actions'])
        left_joint_pos = np.array(demo_group['obs/left_joint_pos'])
        right_joint_pos = np.array(demo_group['obs/right_joint_pos'])
        left_images = np.array(demo_group['obs/left'])
        right_images = np.array(demo_group['obs/right'])
        top_images = np.array(demo_group['obs/top'])
    except KeyError:
        print(f'Demo {demo_name} is not valid and will be skipped.', file=sys.stderr)
        return False

    # Preprocess actions and joint positions
    actions = preprocess_joint_pos(actions)
    left_joint_pos = preprocess_joint_pos(left_joint_pos)
    right_joint_pos = preprocess_joint_pos(right_joint_pos)

    assert actions.shape[0] == left_joint_pos.shape[0] == right_joint_pos.shape[0] == left_images.shape[0] == right_images.shape[0] == top_images.shape[0]
    total_state_frames = actions.shape[0]
    # Skip the first 5 frames to avoid initial artifacts
    for frame_index in range(5, total_state_frames):
        frame = {
            "action": actions[frame_index],
            "observation.state": np.concatenate([left_joint_pos[frame_index], right_joint_pos[frame_index]]),
            "observation.images.left": left_images[frame_index],
            "observation.images.top": top_images[frame_index],
            "observation.images.right": right_images[frame_index],
        }
        # Let the conductor handle episode_index and global index during post-processing
        dataset.add_frame(frame=frame, task=task)

    return True

def scan_demos(hdf5_file):
    """Scans an HDF5 file and prints valid demo names, one per line."""
    tasks = []
    try:
        with h5py.File(hdf5_file, 'r') as f:
            for demo_name in f['data'].keys():
                demo_group = f['data'][demo_name]
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    continue
                tasks.append(demo_name)
    except Exception as e:
        print(f"Could not read demos from {hdf5_file}: {e}", file=sys.stderr)
    
    for task in tasks:
        print(task)

def convert_numpy_to_list(data):
    """Recursively converts numpy arrays in a dictionary or list to native Python lists."""
    if isinstance(data, dict):
        return {k: convert_numpy_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_list(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def main():
    """Main function to handle command-line arguments for the worker process."""
    parser = argparse.ArgumentParser(description='Worker for converting a single Isaac Sim demo to LeRobot format.')
    
    # Mode argument
    parser.add_argument(
        '--scan',
        action='store_true',
        help='If provided, scan the HDF5 file and print demo names, then exit.'
    )

    # All other arguments
    parser.add_argument('--hdf5-file', type=str, required=True, help='Path to the source HDF5 file.')
    parser.add_argument('--demo-name', type=str, help='The name of the demo to process within the HDF5 file.')
    parser.add_argument('--output-dir', type=str, help='The directory to save the temporary dataset.')
    parser.add_argument('--repo-id', type=str, help='The HuggingFace repository ID for metadata purposes.')
    parser.add_argument('--robot-type', type=str, choices=['so101_follower', 'bi_so101_follower'], help='The type of robot configuration.')
    parser.add_argument('--fps', type=int, help='The frames per second for the video.')
    parser.add_argument('--task', type=str, help='A description of the task.')

    args = parser.parse_args()

    if args.scan:
        scan_demos(args.hdf5_file)
        return

    # --- Full conversion mode ---
    if not all([args.demo_name, args.output_dir, args.repo_id, args.robot_type, args.fps, args.task]):
        parser.error('All arguments are required for conversion mode (when --scan is not used).')
        return

    # Import lerobot after parsing args, as it can be slow
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        # Create the dataset in the specified temporary output directory.
        # The repo_id is used for metadata, but the data is stored locally.
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.output_dir,
            fps=args.fps,
            robot_type=args.robot_type,
            features=SINGLE_ARM_FEATURES if args.robot_type == 'so101_follower' else BI_ARM_FEATURES,
        )
        
        with h5py.File(args.hdf5_file, 'r') as f:
            demo_group = f['data'][args.demo_name]
            
            if args.robot_type == 'so101_follower':
                valid = process_single_arm_data(dataset, args.task, demo_group, args.demo_name)
            elif args.robot_type == 'bi_so101_follower':
                valid = process_bi_arm_data(dataset, args.task, demo_group, args.demo_name)
            else:
                valid = False

            if valid:
                dataset.save_episode()

                # The episode is now saved within the temporary worker directory.
                # We need to extract the metadata that the orchestrator will need for the final merge.
                
                # Reload the metadata that was just written to the temp dir
                temp_meta = dataset.meta
                last_episode_index = temp_meta.total_episodes - 1
                
                # Extract the necessary info for the orchestrator
                episode_stats = temp_meta.episodes_stats[last_episode_index]
                episode_info = temp_meta.episodes[last_episode_index]
                
                output_payload = {
                    "root": str(dataset.root),
                    "stats": episode_stats,
                    "tasks": episode_info["tasks"],
                    "length": episode_info["length"],
                }

                # Convert any numpy arrays to lists to ensure it's JSON serializable
                sanitized_payload = convert_numpy_to_list(output_payload)

                # Print the payload as a JSON string for the orchestrator to parse from stdout
                print(json.dumps(sanitized_payload))
                sys.exit(0)

    except Exception as e:
        import traceback
        print(f"Worker for demo {args.demo_name} failed with an exception.", file=sys.stderr)
        print(f"Exception type: {type(e)}", file=sys.stderr)
        print(f"Exception message: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    
    # If the code reaches here, it means 'valid' was False, which is also a failure.
    print(f"Worker for demo {args.demo_name} failed: Data processing returned invalid.", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    main()
