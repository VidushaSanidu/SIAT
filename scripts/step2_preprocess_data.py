#!/usr/bin/env python3
"""
Step 2: Preprocess Dataset

This script converts your ETH/UCY format .txt files into the .npz format
required by the SIAT model. Run this after step1_check_environment.py passes.

Input: .txt files in datasets/ directory (format: frame_id, pedestrian_id, x, y)
Output: .npz files in data_npz/ directory
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
from pathlib import Path


def load_eth_ucy_file(file_path):
    """
    Load ETH/UCY format file.
    
    Expected format: frame_id, pedestrian_id, x, y (tab-separated)
    
    Returns:
        Dictionary mapping pedestrian_id -> trajectory array
    """
    print(f"  📖 Loading {os.path.basename(file_path)}...")
    
    try:
        # Read the file with flexible separator handling
        data = pd.read_csv(file_path, sep='\t', header=None, 
                          names=['frame', 'ped_id', 'x', 'y'])
        
        print(f"     - {len(data)} data points")
        print(f"     - {data['ped_id'].nunique()} unique pedestrians")
        print(f"     - Frame range: {data['frame'].min():.0f} to {data['frame'].max():.0f}")
        
        # Group by pedestrian ID
        trajectories = {}
        for ped_id, group in data.groupby('ped_id'):
            # Sort by frame and extract coordinates
            group_sorted = group.sort_values('frame')
            coords = group_sorted[['x', 'y']].values
            trajectories[int(ped_id)] = coords.astype(np.float32)
        
        # Print trajectory length stats
        traj_lengths = [len(traj) for traj in trajectories.values()]
        print(f"     - Trajectory lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")
        
        return trajectories
        
    except Exception as e:
        print(f"     ❌ Error loading {file_path}: {e}")
        return {}


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
    
    if len(trajectories) < min_agents:
        print(f"     ⚠️  Only {len(trajectories)} agents, need at least {min_agents}")
        return samples
    
    # Convert to list for easier indexing
    ped_ids = list(trajectories.keys())
    ped_trajs = [trajectories[pid] for pid in ped_ids]
    
    # Find common time range where we have enough data
    min_traj_len = min(len(traj) for traj in ped_trajs)
    
    if min_traj_len < total_len:
        print(f"     ⚠️  Trajectories too short ({min_traj_len} < {total_len})")
        return samples
    
    print(f"     - Creating sliding windows (obs={obs_len}, pred={pred_len})")
    
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
    
    print(f"     ✅ Created {len(samples)} training samples")
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
    print(f"\n🔄 Processing {os.path.basename(input_file)}...")
    
    # Load trajectories
    trajectories = load_eth_ucy_file(input_file)
    
    if not trajectories:
        print(f"     ❌ No valid trajectories found")
        return False
    
    # Create samples
    samples = create_sliding_windows(trajectories, obs_len, pred_len)
    
    if len(samples) == 0:
        print(f"     ❌ No valid samples created")
        return False
    
    # Convert to the format expected by TrajectoryDataset
    obs_list = []
    fut_list = []
    window_list = []
    
    for obs, fut, window in samples:
        obs_list.append(obs)
        fut_list.append(fut)
        window_list.append(window)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save as npz
    np.savez_compressed(output_file,
                       observations=np.array(obs_list),
                       futures=np.array(fut_list),
                       windows=np.array(window_list, dtype=object))
    
    print(f"     ✅ Saved {len(samples)} samples to {os.path.basename(output_file)}")
    return True


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
    
    print("🔄 SIAT Data Preprocessing")
    print("=" * 50)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Observation length: {args.obs_len}")
    print(f"Prediction length:  {args.pred_len}")
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory {args.input_dir} does not exist!")
        return False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .txt files
    txt_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    if not txt_files:
        print(f"❌ No .txt files found in {args.input_dir}")
        return False
    
    print(f"\n📊 Found {len(txt_files)} .txt files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for txt_file in txt_files:
        # Create output filename
        rel_path = os.path.relpath(txt_file, args.input_dir)
        npz_name = os.path.splitext(rel_path.replace('/', '_'))[0] + '.npz'
        output_file = os.path.join(args.output_dir, npz_name)
        
        try:
            if preprocess_scene(txt_file, output_file, args.obs_len, args.pred_len):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"     ❌ Error: {e}")
            failed += 1
    
    print(f"\n" + "=" * 50)
    print("📋 PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed to process:      {failed} files")
    print(f"📁 Output directory:       {args.output_dir}")
    
    if successful > 0:
        # List output files
        npz_files = [f for f in os.listdir(args.output_dir) if f.endswith('.npz')]
        print(f"\n📦 Created {len(npz_files)} .npz files:")
        for npz_file in sorted(npz_files):
            file_path = os.path.join(args.output_dir, npz_file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   - {npz_file} ({file_size:.1f} KB)")
        
        print(f"\n🎉 Preprocessing complete!")
        print(f"\n📋 Next Step:")
        print(f"   Run: python scripts/step3_test_compatibility.py")
        
        return True
    else:
        print(f"\n❌ Preprocessing failed! No files were successfully processed.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
