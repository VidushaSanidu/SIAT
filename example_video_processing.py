#!/usr/bin/env python3
"""
Example usage of the video processing pipeline.

This example demonstrates:
1. Processing a single video with YOLO
2. Extracting pedestrian trajectories
3. Creating training data for SIAT model
4. Visualizing results
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video import (
    VideoProcessor,
    extract_trajectories_from_detections,
    visualize_detections
)
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_process_single_video():
    """Example: Process a single video file."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Processing a Single Video")
    logger.info("="*60)
    
    # Initialize video processor with YOLOv8 nano model (fastest)
    processor = VideoProcessor(
        model_name='yolov8n.pt',  # nano model - fastest
        confidence_threshold=0.5,
        device='cuda'  # or 'cpu' if GPU not available
    )
    
    # Process video
    video_path = './videos/sample.avi'  # Replace with your video path
    
    # For demonstration, show the function signature
    logger.info(f"\nCalling: VideoProcessor.process_video('{video_path}')")
    logger.info("Parameters:")
    logger.info("  - sample_rate: 1 (process every frame)")
    logger.info("  - confidence_threshold: 0.5")
    
    # In practice:
    # detections = processor.process_video(video_path, sample_rate=1)
    # This returns: {frame_id: [[x1, y1, x2, y2, conf], ...], ...}


def example_extract_trajectories():
    """Example: Extract trajectories from detections."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Extract Trajectories from Detections")
    logger.info("="*60)
    
    # Simulate some detections (in practice, these come from VideoProcessor)
    logger.info("\nSimulating YOLO detections from 100 frames...")
    
    # Create mock detections: {frame_id: array of [x1, y1, x2, y2, conf]}
    detections = {}
    np.random.seed(42)
    
    for frame_id in range(100):
        n_people = np.random.randint(1, 4)  # 1-3 people per frame
        frame_detections = np.random.rand(n_people, 4) * [640, 480, 640, 480]
        frame_detections[:, 2] = frame_detections[:, 0] + np.random.rand(n_people) * 100
        frame_detections[:, 3] = frame_detections[:, 1] + np.random.rand(n_people) * 150
        detections[frame_id] = frame_detections
    
    logger.info(f"Created detections for {len(detections)} frames")
    
    # Extract trajectories
    logger.info("\nExtracting trajectories using centroid tracking...")
    trajectories = extract_trajectories_from_detections(
        detections,
        max_disappeared=50,
        max_distance=50,
        min_trajectory_length=8  # Minimum 8 frames per trajectory
    )
    
    logger.info(f"Extracted {len(trajectories)} trajectories")
    
    # Print trajectory statistics
    for traj_id, trajectory in list(trajectories.items())[:3]:  # Show first 3
        logger.info(f"\n  Trajectory {traj_id}:")
        logger.info(f"    - Length: {trajectory.length} frames")
        logger.info(f"    - Position range X: [{trajectory.positions[:, 0].min():.1f}, {trajectory.positions[:, 0].max():.1f}]")
        logger.info(f"    - Position range Y: [{trajectory.positions[:, 1].min():.1f}, {trajectory.positions[:, 1].max():.1f}]")


def example_create_training_data():
    """Example: Create training data in SIAT format."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Create Training Data for SIAT")
    logger.info("="*60)
    
    from src.video import TrajectoryExtractor, Trajectory
    
    # Create mock trajectories
    logger.info("\nCreating mock trajectories...")
    trajectories = {}
    
    for ped_id in range(3):
        frames = list(range(20 + ped_id * 5))
        positions = np.cumsum(np.random.randn(len(frames), 2) * 5, axis=0) + np.random.rand(2) * 100
        trajectories[ped_id] = Trajectory(ped_id, frames, positions)
    
    logger.info(f"Created {len(trajectories)} mock trajectories")
    
    # Create training data
    extractor = TrajectoryExtractor()
    extractor.trajectories = trajectories
    
    obs_len = 8
    pred_len = 12
    
    logger.info(f"\nCreating training samples (obs_len={obs_len}, pred_len={pred_len})...")
    
    observations, futures, windows = extractor.create_training_data(
        obs_len=obs_len,
        pred_len=pred_len,
        trajectories=trajectories
    )
    
    logger.info(f"Training data created:")
    logger.info(f"  - Observations shape: {observations.shape}")
    logger.info(f"  - Futures shape: {futures.shape}")
    logger.info(f"  - Number of scene windows: {len(windows)}")
    
    if len(windows) > 0:
        logger.info(f"  - Example window shape: {windows[0].shape}")


def example_model_input_format():
    """Example: Show the expected input format for SIAT model."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: SIAT Model Input Format")
    logger.info("="*60)
    
    config = Config()
    
    logger.info("\nExpected model inputs:")
    logger.info(f"  obs: shape (batch_size, {config.model.obs_len}, 2)")
    logger.info(f"       - Observed trajectory of target pedestrian")
    logger.info(f"       - Example: (32, {config.model.obs_len}, 2) for batch of 32")
    
    logger.info(f"\n  full_window: shape (batch_size, n_agents, {config.model.obs_len + config.model.pred_len}, 2)")
    logger.info(f"       - All pedestrians in scene during obs+pred window")
    logger.info(f"       - n_agents varies per batch item")
    logger.info(f"       - Example: (32, 5, {config.model.obs_len + config.model.pred_len}, 2)")
    
    logger.info(f"\n  agent_mask: shape (batch_size, n_agents)")
    logger.info(f"       - Boolean mask for valid agents")
    logger.info(f"       - Example: (32, 5) with True/False values")
    
    logger.info("\nExpected model output:")
    logger.info(f"  pred: shape (batch_size, {config.model.pred_len}, 2)")
    logger.info(f"       - Predicted trajectory of target pedestrian")
    logger.info(f"       - Example: (32, {config.model.pred_len}, 2)")


def example_complete_pipeline():
    """Example: Complete pipeline from video to model input."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 5: Complete Pipeline")
    logger.info("="*60)
    
    logger.info("""
Complete workflow:

1. VIDEO INPUT
   └─ videos/pedestrian_video.avi
   
2. YOLO DETECTION
   └─ VideoProcessor.process_video()
   └─ Output: detections = {frame_id: [[x1, y1, x2, y2, conf], ...]}
   
3. TRAJECTORY EXTRACTION
   └─ extract_trajectories_from_detections()
   └─ Output: trajectories = {ped_id: Trajectory(...)}
   
4. TRAINING DATA CREATION
   └─ TrajectoryExtractor.create_training_data()
   └─ Output: (observations, futures, windows)
   
5. SAVE TO NPZ
   └─ np.savez(output.npz, observations=obs, futures=fut, windows=win)
   
6. LOAD WITH DATASET
   └─ TrajectoryDataset(npz_files=['output.npz'])
   
7. CREATE BATCHES
   └─ DataLoader(dataset, collate_fn=collate_fn)
   
8. TRAIN SIAT MODEL
   └─ predictions = model(obs, window, agent_mask)
   """)


def main():
    """Run all examples."""
    
    logger.info("\n" + "="*80)
    logger.info("VIDEO PROCESSING PIPELINE - USAGE EXAMPLES")
    logger.info("="*80)
    
    example_process_single_video()
    example_extract_trajectories()
    example_create_training_data()
    example_model_input_format()
    example_complete_pipeline()
    
    logger.info("\n" + "="*80)
    logger.info("QUICK START")
    logger.info("="*80)
    logger.info("""
To process your own videos:

1. Place your .avi files in a directory (e.g., ./videos/)

2. Run the processing script:
   python scripts/step0_process_videos.py \\
       --video_dir ./videos \\
       --output_dir ./data_npz \\
       --model yolov8n.pt \\
       --confidence 0.5 \\
       --device cuda
       
3. This will create .npz files in data_npz/

4. Use with existing training pipeline:
   python scripts/step4_train_model.py

Options:
  --model: yolov8n.pt (nano/fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt (large/slowest)
  --confidence: Detection threshold (0.3-0.7 recommended)
  --sample_rate: Process every nth frame (1=all, 2=every other)
  --max_distance: Centroid matching distance threshold (in pixels)
    """)


if __name__ == '__main__':
    main()
