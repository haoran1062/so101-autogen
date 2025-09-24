import os
import sys
import subprocess
import argparse
import multiprocessing
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
# from tqdm import tqdm # MOVED
# from lerobot.datasets.lerobot_dataset import LeRobotDataset # MOVED
import pprint

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
        "info": {
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
        "info": {
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
        "info": {
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

# Define a chunk size for storing episodes. LeRobot's default is 1000.
CHUNK_SIZE = 1000

"""
This script orchestrates the parallel conversion of HDF5 datasets to the LeRobot
format. It manages worker processes that handle the actual data conversion,
ensuring the main process remains clean and avoids multiprocessing-related issues.
"""

def scan_for_tasks(hdf5_files, worker_script_path, python_executable):
    """Uses the worker script in --scan mode to discover all demos."""
    tasks = []
    print("Scanning HDF5 files to gather all demos...")
    for hdf5_file in hdf5_files:
        try:
            # Use a subprocess to call the worker in scan mode
            result = subprocess.run(
                [python_executable, worker_script_path, '--hdf5-file', hdf5_file, '--scan'],
                capture_output=True,
                text=True,
                check=True
            )
            # The worker prints one demo name per line
            demo_names = result.stdout.strip().split('\n')
            for demo_name in demo_names:
                if demo_name: # Avoid empty lines
                    tasks.append((hdf5_file, demo_name))
        except subprocess.CalledProcessError as e:
            print(f"Failed to scan {hdf5_file}: {e.stderr}", file=sys.stderr)
        except FileNotFoundError:
            print(f"Error: Worker script not found at '{worker_script_path}'", file=sys.stderr)
            sys.exit(1)
            
    return tasks

def run_worker_subprocess(args_tuple):
    """Function executed by the multiprocessing pool to call the worker script."""
    (worker_script_path, hdf5_file_path, demo_name, output_dir, 
     repo_id, robot_type, fps, task, worker_idx, python_executable) = args_tuple
    
    try:
        # Construct the command to run the worker script
        command = [
            python_executable,
            worker_script_path,
            '--hdf5-file', hdf5_file_path,
            '--demo-name', demo_name,
            '--output-dir', output_dir,
            '--repo-id', repo_id,
            '--robot-type', robot_type,
            '--fps', str(fps),
            '--task', task,
        ]
        
        # We don't capture output in real-time here to allow workers to print errors directly
        # to their own logs, which we'll write after the process completes.
        result = subprocess.run(command, check=False, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        
        # Now that the worker has run (and created its directory), we can write the logs.
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'stdout.txt'), 'w') as f:
            f.write(result.stdout)
        with open(os.path.join(output_dir, 'stderr.txt'), 'w') as f:
            f.write(result.stderr)

        if result.returncode == 0:
            # On success, return the original demo_name along with the output
            return True, demo_name, result.stdout.strip(), output_dir
        else:
            print(f"\n--- Worker {worker_idx} for demo '{demo_name}' failed. ---")
            print(f"Stderr:\n{result.stderr}")
            print(f"Stdout:\n{result.stdout}")
            print(f"--------------------------------------------------")
            return False, demo_name, None, None
    except Exception as e:
        print(f"\n--- Worker {worker_idx} for demo '{demo_name}' failed with an exception in the conductor. ---")
        print(f"Exception: {e}")
        print(f"------------------------------------------------------------------------------------")
        return False, demo_name, None, None


def main():
    """Main function that parses arguments and orchestrates the conversion process."""
    parser = argparse.ArgumentParser(
        description='Convert an Isaac Sim HDF5 dataset to the LeRobot dataset format in parallel.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- User-facing arguments ---
    parser.add_argument('--repo-id',type=str,default='matrix/so101_sync_orange_pick',help='The HuggingFace repository ID for the dataset.')
    parser.add_argument('--robot-type',type=str,choices=['so101_follower', 'bi_so101_follower'],default='so101_follower',help='The type of robot configuration.')
    parser.add_argument('--fps',type=int,default=30,help='The frames per second for the dataset videos.')
    parser.add_argument('--hdf5-root',type=str,default='./datasets',help='The root directory containing the source HDF5 files.')
    parser.add_argument('--hdf5-files',type=str,nargs='+',default=['dataset.hdf5'],help='A list of HDF5 files to process (relative to hdf5-root).')
    parser.add_argument('--task',type=str,default='Grab orange and place into plate',help='A description of the task being performed in the dataset.')
    parser.add_argument('--push-to-hub',action='store_true',help='Push the converted dataset to the HuggingFace Hub upon completion.')
    parser.add_argument('--num-workers',type=int,default=max(1, multiprocessing.cpu_count() // 2),help='The number of parallel workers to use for conversion.')
    parser.add_argument(
        '--python-executable',
        type=str,
        default=sys.executable,
        help='The python executable to use for the worker subprocesses.'
    )
    parser.add_argument(
        '--intermediate-dir',
        type=str,
        default=None,
        help='If provided, skip the processing step and use this directory for merging worker outputs.'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='If provided, only convert the first N episodes found across all HDF5 files.'
    )

    args = parser.parse_args()
    
    # --- Script paths and file lists ---
    # Construct an absolute path to the worker script, assuming execution from the project root
    worker_script_path = os.path.abspath(os.path.join('scripts', 'convert_worker.py'))
    hdf5_files = [os.path.join(args.hdf5_root, f) for f in args.hdf5_files]
    
    # --- Start Debug Prints ---
    print("--- DEBUG INFO ---")
    print(f"Current CWD: {os.getcwd()}")
    print(f"Resolved worker script path: {worker_script_path}")
    print(f"Worker script exists at path: {os.path.exists(worker_script_path)}")
    print(f"Python executable for subprocess: {args.python_executable}")
    print("--- END DEBUG INFO ---\n")

    print("--- Conversion Parameters ---")
    for key, value in vars(args).items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print("---------------------------\n")


    all_worker_dirs = []
    successful_workers_outputs = []
    temp_dir = None

    # --- Clean up previous final dataset directory if it exists ---
    # Manually construct the path to avoid depending on a specific lerobot version's API.
    hf_cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    lerobot_cache_home = os.path.join(hf_cache_home, "lerobot")
    final_dataset_path = os.path.join(lerobot_cache_home, args.repo_id)

    if os.path.exists(final_dataset_path):
        print(f"Removing existing final dataset directory: {final_dataset_path}")
        shutil.rmtree(final_dataset_path)

    if args.intermediate_dir:
        print(f"--- Resuming from intermediate directory: {args.intermediate_dir} ---")
        if not os.path.isdir(args.intermediate_dir):
            print(f"Error: Intermediate directory not found at {args.intermediate_dir}")
            return
        
        # In resume mode, we need to find the worker outputs (stdout.txt) and parse them
        for worker_dir_name in os.listdir(args.intermediate_dir):
            worker_dir_path = os.path.join(args.intermediate_dir, worker_dir_name)
            if os.path.isdir(worker_dir_path):
                all_worker_dirs.append(worker_dir_path)
                stdout_path = os.path.join(worker_dir_path, 'stdout.txt')
                if os.path.exists(stdout_path):
                    with open(stdout_path, 'r') as f:
                        output = f.read().strip()
                        if output:
                            successful_workers_outputs.append(output)
        
        print(f"Found {len(successful_workers_outputs)} successful worker outputs to merge.")

    else:
        # --- Step 1: Scan for all demos to process ---
        print("Scanning HDF5 files to gather all demos...")
        tasks = scan_for_tasks(hdf5_files, worker_script_path, args.python_executable)
        if not tasks:
            print("No valid demos found to process. Exiting.")
            return
        
        # Apply the limit if provided
        if args.limit is not None:
            if args.limit <= 0:
                print("Error: --limit must be a positive integer.")
                return
            tasks = tasks[:args.limit]
            print(f"--- Limiting conversion to the first {len(tasks)} episodes ---")

        print(f"Found a total of {len(tasks)} demos to process.\n")

        # --- Step 2: Prepare for parallel execution ---
        main_temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory for worker outputs: {main_temp_dir}\n")
        temp_dir = main_temp_dir # Store for cleanup

        worker_args = []
        for i, (hdf5_file_path, demo_name) in enumerate(tasks):
            worker_output_dir = os.path.join(main_temp_dir, f"worker_{i}")
            # The worker script will create this directory.
            worker_args.append(
                (worker_script_path, hdf5_file_path, demo_name, worker_output_dir, 
                 args.repo_id, args.robot_type, args.fps, args.task, i, args.python_executable)
            )

        # --- Step 3: Run workers in parallel ---
        print(f"Starting conversion with {args.num_workers} workers...")
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            from tqdm import tqdm
            with tqdm(total=len(tasks), desc="Processing demos") as pbar:
                # Store results in a dictionary keyed by demo_name to preserve order
                results = {}
                for success, demo_name, output_json, output_dir in pool.imap_unordered(run_worker_subprocess, worker_args):
                    if success:
                        # Store both the JSON payload and the path to the worker's output directory
                        results[demo_name] = (output_json, output_dir)
                    pbar.update()

        # Sort the successful outputs based on the original task order
        # The list will contain tuples of (demo_name, (output_json, output_dir))
        successful_workers_info = []
        for _, demo_name in tasks:
            if demo_name in results:
                successful_workers_info.append((demo_name, results[demo_name]))
        
        if not successful_workers_info:
            print("\nNo demos were processed successfully. Exiting.", file=sys.stderr)
            if temp_dir:
                shutil.rmtree(temp_dir)
            return
    
        print(f"\nSuccessfully processed {len(successful_workers_info)} demos.\n")


    # --- Step 4: Merge results from workers ---
    print("All workers finished. Merging results...")

    # We can only import this now, after all multiprocessing is done.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # --- Create the final, empty dataset ---
    # The final dataset will be created in the local cache, determined by the repo_id.
    # We create it once here, and then add episodes to it.
    print(f"Creating final dataset shell for '{args.repo_id}'...")
    final_dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=SINGLE_ARM_FEATURES if args.robot_type == 'so101_follower' else BI_ARM_FEATURES,
    )
    print(f"Final dataset created at: {final_dataset.root}")
    
    # Create the required subdirectories
    meta_dir = os.path.join(final_dataset.root, 'meta')
    final_data_dir = os.path.join(final_dataset.root, 'data')
    final_videos_dir = os.path.join(final_dataset.root, 'videos')
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(final_data_dir, exist_ok=True)
    os.makedirs(final_videos_dir, exist_ok=True)

    # Manually initialize the metadata lists, as they are dicts by default on a fresh dataset.
    final_dataset.meta.episodes_stats = []
    final_dataset.meta.episodes = []

    # --- Manually merge all episodes from all successful workers IN A SINGLE LOOP ---
    all_tasks = set()
    global_frame_offset = 0
    from tqdm import tqdm
    # Sort by demo_name to ensure deterministic order
    sorted_worker_info = sorted(successful_workers_info)

    for episode_idx, (demo_name, (worker_output_json, worker_output_dir)) in enumerate(tqdm(sorted_worker_info, desc="Merging and Correcting")):
        try:
            # --- Part A: Handle Metadata Correction ---
            worker_payload = json.loads(worker_output_json)
            
            # 1. Start with the raw statistics from the worker, which has the correct format
            worker_stats = { "episode_index": episode_idx, "stats": worker_payload["stats"] }
            
            # 2. Correct episode_index stats
            worker_stats['stats']['episode_index']['min'] = [episode_idx]
            worker_stats['stats']['episode_index']['max'] = [episode_idx]
            worker_stats['stats']['episode_index']['mean'] = [float(episode_idx)]
            worker_stats['stats']['episode_index']['std'] = [0.0]

            # 3. Correct global index stats
            num_frames = worker_stats['stats']['index']['count'][0]
            min_index = global_frame_offset
            max_index = global_frame_offset + num_frames - 1
            worker_stats['stats']['index']['min'] = [min_index]
            worker_stats['stats']['index']['max'] = [max_index]
            worker_stats['stats']['index']['mean'] = [(min_index + max_index) / 2.0]
            if 'frame_index' in worker_stats['stats']:
                 worker_stats['stats']['index']['std'] = worker_stats['stats']['frame_index']['std']
            
            final_dataset.meta.episodes_stats.append(worker_stats)

            # 4. Store the main episode info
            final_dataset.meta.episodes.append({
                "episode_index": episode_idx,
                "length": worker_payload["length"],
                "tasks": worker_payload["tasks"],
            })
            for task_name in worker_payload["tasks"]:
                all_tasks.add(task_name)

            # --- Part B: Find and move data/video files ---
            worker_root = worker_payload["root"]
            found_parquet = None
            found_videos = []

            for dirpath, _, filenames in os.walk(worker_root):
                for filename in filenames:
                    if filename.endswith('.parquet'):
                        found_parquet = os.path.join(dirpath, filename)
                    elif filename.endswith('.mp4'):
                        camera_name = os.path.basename(dirpath)
                        found_videos.append({"path": os.path.join(dirpath, filename), "camera": camera_name})
            
            # 3. Move and correct the data file (.parquet)
            if found_parquet:
                chunk_idx = episode_idx // CHUNK_SIZE
                chunk_dir = os.path.join(final_data_dir, f'chunk-{chunk_idx:03d}')
                os.makedirs(chunk_dir, exist_ok=True)
                
                episode_filename = f'episode_{episode_idx:06d}.parquet'
                dest_parquet = os.path.join(chunk_dir, episode_filename)
                shutil.move(found_parquet, dest_parquet)

                # Correct the parquet file in its final destination
                df = pd.read_parquet(dest_parquet)
                df['episode_index'] = episode_idx
                df['index'] = np.arange(global_frame_offset, global_frame_offset + len(df))
                df.to_parquet(dest_parquet)
            else:
                print(f"Warning: No .parquet file found for worker output: {worker_root}")
                continue

            # 4. Move the video files (.mp4)
            for video_info in found_videos:
                chunk_idx = episode_idx // CHUNK_SIZE
                chunk_dir = os.path.join(final_videos_dir, f'chunk-{chunk_idx:03d}')
                final_cam_dir = os.path.join(chunk_dir, video_info["camera"])
                os.makedirs(final_cam_dir, exist_ok=True)

                episode_filename = f'episode_{episode_idx:06d}.mp4'
                dest_video = os.path.join(final_cam_dir, episode_filename)
                shutil.move(video_info["path"], dest_video)
            
            # 6. Update the global offset for the next episode
            global_frame_offset += worker_payload["length"]

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse worker output. Skipping. Error: {e}. Output: '{worker_output_json}'")

    # --- Write the final aggregated metadata to disk ---
    # Manually write the .jsonl files to the 'meta' directory
    with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w") as f:
        for stats in final_dataset.meta.episodes_stats:
            f.write(json.dumps(stats) + "\n")
            
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for episode_info in final_dataset.meta.episodes:
            f.write(json.dumps(episode_info) + "\n")

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for i, task_name in enumerate(sorted(list(all_tasks))):
            f.write(json.dumps({"task_index": i, "task": task_name}) + "\n")

    # --- Manually create and write the final info.json ---
    num_episodes = len(final_dataset.meta.episodes)
    num_cameras = len(SINGLE_ARM_FEATURES) - 2 # Subtract action and state
    
    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": num_episodes,
        "total_frames": sum(e["length"] for e in final_dataset.meta.episodes),
        "total_tasks": len(all_tasks),
        "total_videos": num_episodes * num_cameras,
        "total_chunks": (num_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": float(args.fps),
        "splits": {
            "train": f"0:{num_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": final_dataset.meta.info["features"]
    }

    # Add the duplicated 'info' key to video features, mimicking the reference file
    for key, value in info["features"].items():
        if value["dtype"] == "video":
            # Ensure fps is a float in the feature definition as well
            if "video.fps" in value["video_info"]:
                value["video_info"]["video.fps"] = float(value["video_info"]["video.fps"])
            value["info"] = value["video_info"]

    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


    print(f"\nSuccessfully merged {len(final_dataset.meta.episodes)} episodes into {final_dataset.root}\n")

    # --- Step 5: Push to Hub (if requested) ---
    if args.push_to_hub:
        print("Pushing dataset to HuggingFace Hub...")
        # The dataset now exists locally, so we can just load and push.
        final_dataset.push_to_hub()
        print("Push to hub complete.")

    # --- Step 6: Cleanup ---
    if temp_dir:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("Cleanup complete.")

if __name__ == '__main__':
    # Using the default start method is fine now, as this script is clean.
    # 'fork' is the default on Linux and is efficient.
    main()
