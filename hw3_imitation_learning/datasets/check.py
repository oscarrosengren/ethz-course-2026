import os
import zarr
import numpy as np

def check_datasets():
    base_path = 'datasets/raw/multi_cube/teleop'
    
    # Print Header
    print(f"\n{'Folder Name':<25} | {'Episodes':<10} | {'Total Steps':<12} | Status")
    print("-" * 70)

    if not os.path.exists(base_path):
        print(f"Error: Path '{base_path}' not found. Are you in the right directory?")
        return

    # Get all subdirectories and sort them by time
    folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    for folder in folders:
        zarr_path = os.path.join(base_path, folder, 'so100_multicube_teleop.zarr')
        
        if not os.path.exists(zarr_path):
            # This folder exists but doesn't have a zarr store inside
            continue

        try:
            # Open the zarr store in read-only mode
            data = zarr.open(zarr_path, mode='r')
            
            # Find the episode_ends array (sometimes it's in meta/, sometimes top level)
            if 'meta' in data and 'episode_ends' in data['meta']:
                ends = data['meta/episode_ends'][:]
            elif 'episode_ends' in data:
                ends = data['episode_ends'][:]
            else:
                print(f"{folder:<25} | ❌ Error    | {'--':<12} | Missing episode_ends")
                continue

            num_episodes = len(ends)
            # Total steps is the value of the last index in episode_ends
            total_steps = int(ends[-1]) if num_episodes > 0 else 0
            
            if num_episodes > 0:
                status = "✅ READY"
                print(f"{folder:<25} | {num_episodes:<10} | {total_steps:<12} | {status}")
            else:
                status = "⚪ EMPTY"
                print(f"{folder:<25} | {num_episodes:<10} | {total_steps:<12} | {status}")

        except Exception as e:
            # Catch version mismatches or corrupted files
            print(f"{folder:<25} | ⚠️  LOCKED   | {'--':<12} | Corrupted or Zarr version error")

    print("-" * 70 + "\n")

if __name__ == "__main__":
    check_datasets()