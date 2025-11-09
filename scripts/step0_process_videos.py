#!/usr/bin/env python3
"""
Step 0: Process video files to extract pedestrian trajectories.

This script:
1. Reads video files (.avi, .mp4, etc.)
2. Detects pedestrians using YOLOv8
3. Tracks pedestrians across frames
4. Extracts trajectories
5. Converts trajectories to model input format (observations, windows)

Usage:
    python step0_process_videos.py --video_dir ./videos --output_dir ./data_npz
"""

import argparse
import logging
import numpy as np
from pathlib import Path
import sys
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video import VideoProcessor, extract_trajectories_from_detections, Trajectory
from src.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_videos_to_trajectories(video_dir: str, 
                                  model_name: str = 'yolov8n.pt',
                                  confidence_threshold: float = 0.5,
                                  device: str = 'cuda',
                                  sample_rate: int = 1,
                                  max_disappeared: int = 50,
                                  max_distance: float = 50) -> Dict[str, Dict[int, Trajectory]]:
    """
    Process video files and extract trajectories.
    
    Args:
        video_dir: Directory containing video files
        model_name: YOLOv8 model name
        confidence_threshold: Detection confidence threshold
        device: Device to run on ('cuda' or 'cpu')
        sample_rate: Process every nth frame
        max_disappeared: Max frames for tracking
        max_distance: Max distance for centroid matching
        
    Returns:
        Dictionary mapping video names to trajectory dictionaries
    """
    video_processor = VideoProcessor(model_name=model_name, 
                                    confidence_threshold=confidence_threshold,
                                    device=device)
    
    all_video_trajectories = {}
    
    # Process all videos in directory
    video_files = sorted(Path(video_dir).glob('*.avi')) + \
                 sorted(Path(video_dir).glob('*.mp4')) + \
                 sorted(Path(video_dir).glob('*.mov'))
    
    if len(video_files) == 0:
        logger.warning(f"No video files found in {video_dir}")
        return {}
    
    logger.info(f"Found {len(video_files)} video files")
    
    for video_file in video_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {video_file.name}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Detect pedestrians in video
            logger.info("Running pedestrian detection...")
            detections = video_processor.process_video(str(video_file), sample_rate=sample_rate)
            
            if len(detections) == 0:
                logger.warning(f"No pedestrians detected in {video_file.name}")
                continue
            
            logger.info(f"Found detections in {len(detections)} frames")
            
            # Step 2: Extract trajectories from detections
            logger.info("Extracting trajectories...")
            trajectories = extract_trajectories_from_detections(
                detections,
                max_disappeared=max_disappeared,
                max_distance=max_distance,
                min_trajectory_length=8  # At least obs_len
            )
            
            if len(trajectories) == 0:
                logger.warning(f"No valid trajectories extracted from {video_file.name}")
                continue
            
            logger.info(f"Extracted {len(trajectories)} trajectories")
            
            # Log trajectory statistics
            trajectory_lengths = [traj.length for traj in trajectories.values()]
            logger.info(f"  Min length: {min(trajectory_lengths)} frames")
            logger.info(f"  Max length: {max(trajectory_lengths)} frames")
            logger.info(f"  Mean length: {np.mean(trajectory_lengths):.1f} frames")
            
            all_video_trajectories[video_file.name] = trajectories
            
        except Exception as e:
            logger.error(f"Error processing {video_file.name}: {e}", exc_info=True)
            continue
    
    return all_video_trajectories


def trajectories_to_npz(video_trajectories: Dict[str, Dict[int, Trajectory]],
                       output_dir: str,
                       obs_len: int = 8,
                       pred_len: int = 12) -> None:
    """
    Convert trajectories to NPZ format compatible with SIAT model.
    
    Args:
        video_trajectories: Dictionary of trajectories per video
        output_dir: Output directory for NPZ files
        obs_len: Observation length
        pred_len: Prediction length
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for video_name, trajectories in video_trajectories.items():
        logger.info(f"\nConverting trajectories from {video_name} to NPZ format...")
        
        observations = []
        futures = []
        windows = []
        
        # Get all trajectory objects
        traj_list = list(trajectories.values())
        
        # For each trajectory, create training samples
        for target_idx, target_traj in enumerate(traj_list):
            # Skip trajectories that are too short
            if target_traj.length < obs_len + pred_len:
                continue
            
            # Slide window through trajectory
            for start_idx in range(target_traj.length - obs_len - pred_len + 1):
                obs = target_traj.positions[start_idx:start_idx + obs_len].copy()
                fut = target_traj.positions[start_idx + obs_len:start_idx + obs_len + pred_len].copy()
                
                # Build scene window
                target_frame_set = set(target_traj.frames[start_idx:start_idx + obs_len + pred_len])
                
                # Collect all agents in this temporal window
                agents_in_window = []
                agents_in_window.append(target_traj.positions[start_idx:start_idx + obs_len + pred_len].copy())
                
                # Add other agents that appear in this time window
                for other_idx, other_traj in enumerate(traj_list):
                    if other_idx == target_idx:
                        continue
                    
                    # Check for temporal overlap
                    other_frame_set = set(other_traj.frames) & target_frame_set
                    if len(other_frame_set) == 0:
                        continue
                    
                    # Extract positions for the window
                    window_pos = []
                    for step in range(obs_len + pred_len):
                        target_frame = target_traj.frames[start_idx + step]
                        if target_frame in other_traj.frames:
                            other_idx_in_traj = other_traj.frames.index(target_frame)
                            window_pos.append(other_traj.positions[other_idx_in_traj].copy())
                        else:
                            # Use last known position if not available
                            if len(window_pos) > 0:
                                window_pos.append(window_pos[-1])
                    
                    if len(window_pos) == obs_len + pred_len:
                        agents_in_window.append(np.array(window_pos))
                
                # Create window array (pad with zeros)
                if len(agents_in_window) > 0:
                    n_agents = len(agents_in_window)
                    window_array = np.zeros((n_agents, obs_len + pred_len, 2), dtype=np.float32)
                    
                    for agent_idx, agent_positions in enumerate(agents_in_window):
                        if len(agent_positions) == obs_len + pred_len:
                            window_array[agent_idx] = agent_positions.astype(np.float32)
                    
                    observations.append(obs.astype(np.float32))
                    futures.append(fut.astype(np.float32))
                    windows.append(window_array)
        
        if len(observations) == 0:
            logger.warning(f"No valid training samples from {video_name}")
            continue
        
        # Save to NPZ
        output_file = output_path / f"video_{video_name.split('.')[0]}.npz"
        np.savez_compressed(
            output_file,
            observations=np.array(observations),
            futures=np.array(futures),
            windows=np.array(windows, dtype=object)  # Store as objects due to variable sizes
        )
        
        logger.info(f"Saved {len(observations)} samples to {output_file}")
        logger.info(f"  Observations shape: {np.array(observations).shape}")
        logger.info(f"  Futures shape: {np.array(futures).shape}")
        logger.info(f"  Windows count: {len(windows)}")


def main():
    parser = argparse.ArgumentParser(
        description='Process video files to extract pedestrian trajectories'
    )
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='./data_npz',
                       help='Output directory for NPZ files')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                       help='YOLOv8 model size (n=nano, s=small, m=medium, l=large)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--sample_rate', type=int, default=1,
                       help='Process every nth frame (1=all frames, 2=every other, etc)')
    parser.add_argument('--max_disappeared', type=int, default=50,
                       help='Max frames a pedestrian can be missing before removing track')
    parser.add_argument('--max_distance', type=float, default=50,
                       help='Max distance (pixels) for centroid matching')
    parser.add_argument('--obs_len', type=int, default=8,
                       help='Observation length')
    parser.add_argument('--pred_len', type=int, default=12,
                       help='Prediction length')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Video Processing Pipeline")
    logger.info("="*60)
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Confidence threshold: {args.confidence}")
    logger.info(f"Device: {args.device}")
    
    # Step 1: Process videos to get trajectories
    logger.info("\n[STEP 1] Processing videos and extracting trajectories...")
    video_trajectories = process_videos_to_trajectories(
        video_dir=args.video_dir,
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
        sample_rate=args.sample_rate,
        max_disappeared=args.max_disappeared,
        max_distance=args.max_distance
    )
    
    if len(video_trajectories) == 0:
        logger.error("No trajectories extracted from videos")
        return 1
    
    # Step 2: Convert trajectories to NPZ format
    logger.info("\n[STEP 2] Converting trajectories to NPZ format...")
    trajectories_to_npz(
        video_trajectories=video_trajectories,
        output_dir=args.output_dir,
        obs_len=args.obs_len,
        pred_len=args.pred_len
    )
    
    logger.info("\n" + "="*60)
    logger.info("Video processing complete!")
    logger.info(f"Output files saved to: {args.output_dir}")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
