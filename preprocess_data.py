"""
Data preprocessing script to convert ETH/UCY format to SIAT-compatible format.

This script converts the tab-separated trajectory files into .npz format
that the TrajectoryDataset class expects.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse


def load_eth_ucy_file(file_path):
    """
    Load ETH/UCY format file.
    
    Expected format: frame_id, pedestrian_id, x, y
    
    Returns:
        Dictionary mapping pedestrian_id -> trajectory array
    """
    # Read the file
    data = pd.read_csv(file_path, sep='\t', header=None, 
                      names=['frame', 'ped_id', 'x', 'y'])
    
    # Group by pedestrian ID
    trajectories = {}
    for ped_id, group in data.groupby('ped_id'):
        # Sort by frame and extract coordinates
        group_sorted = group.sort_values('frame')
        coords = group_sorted[['x', 'y']].values
        trajectories[int(ped_id)] = coords.astype(np.float32)
    
    return trajectories


def create_sliding_windows(trajectories, obs_len=8, pred_len=12, min_agents=2):
    """
    Create sliding windows for trajectory prediction.
    
    Args:
        trajectories: Dict mapping ped_id -> trajectory
        obs_len: Number of observed timesteps
        pred_len: Number of predicted timesteps
        min_agents: Minimum number of agents required in a scene
        
    Returns:
        List of (target_obs, target_fut, all_agents_window) tuples
    """
    samples = []
    total_len = obs_len + pred_len
    
    # Convert to list for easier indexing
    ped_ids = list(trajectories.keys())
    ped_trajs = [trajectories[pid] for pid in ped_ids]
    
    # Find common time range where we have enough agents
    min_traj_len = min(len(traj) for traj in ped_trajs)
    
    if min_traj_len < total_len:
        print(f"Warning: Trajectories too short ({min_traj_len} < {total_len})")
        return samples
    
    # Create sliding windows
    for start_frame in range(min_traj_len - total_len + 1):
        end_frame = start_frame + total_len
        
        # Extract window for all agents
        window_data = []
        valid_agents = []
        
        for i, traj in enumerate(ped_trajs):
            if len(traj) >= end_frame:
                window_data.append(traj[start_frame:end_frame])
                valid_agents.append(ped_ids[i])
        
        if len(window_data) < min_agents:
            continue
            
        # Convert to numpy array
        window_array = np.array(window_data)  # Shape: (N_agents, total_len, 2)
        
        # Create samples for each agent as target
        for target_idx in range(len(window_data)):
            target_obs = window_array[target_idx, :obs_len]
            target_fut = window_array[target_idx, obs_len:]
            
            # Reorder so target agent is at index 0
            reordered_window = np.zeros_like(window_array)
            reordered_window[0] = window_array[target_idx]  # Target at index 0
            
            # Add other agents
            other_idx = 1
            for i in range(len(window_data)):
                if i != target_idx:
                    reordered_window[other_idx] = window_array[i]
                    other_idx += 1
            
            # Only keep the agents we actually have
            final_window = reordered_window[:len(window_data)]
            
            samples.append((target_obs, target_fut, final_window))
    
    return samples


def preprocess_scene(input_file, output_file, obs_len=8, pred_len=12):
    """
    Preprocess a single scene file.
    
    Args:
        input_file: Path to input .txt file
        output_file: Path to output .npz file
        obs_len: Number of observed timesteps
        pred_len: Number of predicted timesteps
    """
    print(f"Processing {input_file}...")
    
    # Load trajectories
    trajectories = load_eth_ucy_file(input_file)
    print(f"  Found {len(trajectories)} pedestrians")
    
    # Create samples
    samples = create_sliding_windows(trajectories, obs_len, pred_len)
    print(f"  Created {len(samples)} samples")
    
    if len(samples) == 0:
        print(f"  Warning: No valid samples created for {input_file}")
        return
    
    # Convert to the format expected by TrajectoryDataset
    obs_list = []
    fut_list = []
    window_list = []
    
    for obs, fut, window in samples:
        obs_list.append(obs)
        fut_list.append(fut)
        window_list.append(window)
    
    # Save as npz
    np.savez(output_file,
             observations=np.array(obs_list),
             futures=np.array(fut_list),
             windows=np.array(window_list, dtype=object))  # object dtype for variable N_agents
    
    print(f"  Saved to {output_file}")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess ETH/UCY data for SIAT')
    parser.add_argument('--input_dir', type=str, default='./datasets',
                       help='Input directory containing .txt files')
    parser.add_argument('--output_dir', type=str, default='./data_npz',
                       help='Output directory for .npz files')
    parser.add_argument('--obs_len', type=int, default=8,
                       help='Number of observed timesteps')
    parser.add_argument('--pred_len', type=int, default=12,
                       help='Number of predicted timesteps')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .txt files
    txt_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    print(f"Found {len(txt_files)} .txt files to process")
    
    # Process each file
    for txt_file in txt_files:
        # Create output filename
        rel_path = os.path.relpath(txt_file, args.input_dir)
        npz_name = os.path.splitext(rel_path.replace('/', '_'))[0] + '.npz'
        output_file = os.path.join(args.output_dir, npz_name)
        
        try:
            preprocess_scene(txt_file, output_file, args.obs_len, args.pred_len)
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
    
    print(f"\nPreprocessing complete! Output saved to {args.output_dir}")


if __name__ == '__main__':
    main()
