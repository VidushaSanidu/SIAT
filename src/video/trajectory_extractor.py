"""
Trajectory extraction module to convert YOLO detections into pedestrian trajectories.
Uses centroid tracking to associate detections across frames and generate trajectories.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Container for a pedestrian trajectory."""
    pedestrian_id: int
    frames: List[int]
    positions: np.ndarray  # Shape: (num_frames, 2) with x,y coordinates
    
    @property
    def length(self) -> int:
        return len(self.frames)
    
    def get_segment(self, start_frame: int, end_frame: int) -> Optional[np.ndarray]:
        """Get trajectory segment between two frames."""
        start_idx = None
        end_idx = None
        
        for i, f in enumerate(self.frames):
            if f == start_frame:
                start_idx = i
            if f == end_frame:
                end_idx = i + 1
        
        if start_idx is None or end_idx is None or start_idx >= end_idx:
            return None
        
        return self.positions[start_idx:end_idx]


class CentroidTracker:
    """
    Simple centroid-based tracker for associating detections across frames.
    
    This tracker works by:
    1. Computing centroids of bounding boxes
    2. Computing distance between current and previous centroids
    3. Matching detections to existing tracks based on distance threshold
    """
    
    def __init__(self, max_disappeared: int = 50, max_distance: float = 50):
        """
        Initialize centroid tracker.
        
        Args:
            max_disappeared: Maximum frames a person can be missing before removing track
            max_distance: Maximum distance (pixels) to consider for matching detections
        """
        self.next_object_id = 0
        self.objects = {}  # id -> centroid position
        self.disappeared = {}  # id -> frame count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def update(self, detections: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of shape (N, 4) with [x1, y1, x2, y2]
            
        Returns:
            Dictionary mapping object IDs to centroids
        """
        # Handle empty detections
        if len(detections) == 0:
            # Mark all objects as disappeared
            self.disappeared = {oid: self.disappeared.get(oid, 0) + 1 
                              for oid in self.objects.keys()}
            # Remove lost objects
            self.objects = {oid: centroid for oid, centroid in self.objects.items()
                          if self.disappeared[oid] <= self.max_disappeared}
            return self.objects
        
        # Compute centroids of new detections
        input_centroids = self._get_centroids(detections)
        
        # If no objects tracked yet, register all detections
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.objects[self.next_object_id] = centroid
                self.disappeared[self.next_object_id] = 0
                self.next_object_id += 1
        else:
            # Match detections to objects
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid] for oid in object_ids])
            
            # Compute distances between current objects and new detections
            D = cdist(object_centroids, input_centroids)
            
            # Match detections to objects using greedy assignment
            rows = D.min(axis=1).argsort()  # Sort by minimum distance
            cols = D[rows, :].argmin(axis=1)
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Register unmatched detections
            unused_cols = set(range(0, input_centroids.shape[0])) - used_cols
            for col in unused_cols:
                self.objects[self.next_object_id] = input_centroids[col]
                self.disappeared[self.next_object_id] = 0
                self.next_object_id += 1
            
            # Mark unmatched objects as disappeared
            unused_rows = set(range(0, len(object_ids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
            
            # Remove lost objects
            self.objects = {oid: centroid for oid, centroid in self.objects.items()
                          if self.disappeared[oid] <= self.max_disappeared}
        
        return self.objects
    
    @staticmethod
    def _get_centroids(detections: np.ndarray) -> np.ndarray:
        """Compute centroids from bounding boxes."""
        cX = (detections[:, 0] + detections[:, 2]) / 2
        cY = (detections[:, 1] + detections[:, 3]) / 2
        return np.column_stack([cX, cY])


class TrajectoryExtractor:
    """
    Extract pedestrian trajectories from frame-by-frame detections.
    Uses centroid tracking to create continuous trajectories.
    """
    
    def __init__(self, max_disappeared: int = 50, max_distance: float = 50, 
                 min_trajectory_length: int = 2):
        """
        Initialize trajectory extractor.
        
        Args:
            max_disappeared: Max frames a person can be missing before removing track
            max_distance: Max distance for matching detections across frames
            min_trajectory_length: Minimum frames required to keep a trajectory
        """
        self.tracker = CentroidTracker(max_disappeared=max_disappeared, 
                                       max_distance=max_distance)
        self.min_trajectory_length = min_trajectory_length
        self.trajectories = {}  # id -> Trajectory object
    
    def process_frame(self, frame_id: int, detections: np.ndarray) -> None:
        """
        Process a single frame of detections.
        
        Args:
            frame_id: Frame number/timestamp
            detections: Array of shape (N, 4+) with at least [x1, y1, x2, y2]
        """
        # Track using centroid tracker
        tracked_objects = self.tracker.update(detections[:, :4])
        
        # Update trajectories
        for object_id, centroid in tracked_objects.items():
            if object_id not in self.trajectories:
                self.trajectories[object_id] = Trajectory(
                    pedestrian_id=object_id,
                    frames=[frame_id],
                    positions=np.array([centroid])
                )
            else:
                self.trajectories[object_id].frames.append(frame_id)
                self.trajectories[object_id].positions = np.vstack([
                    self.trajectories[object_id].positions,
                    centroid
                ])
    
    def get_trajectories(self) -> Dict[int, Trajectory]:
        """
        Get all extracted trajectories.
        
        Returns:
            Dictionary of trajectories filtered by minimum length
        """
        valid_trajectories = {
            tid: traj for tid, traj in self.trajectories.items()
            if traj.length >= self.min_trajectory_length
        }
        return valid_trajectories
    
    def create_training_data(self, obs_len: int = 8, pred_len: int = 12, 
                            trajectories: Optional[Dict[int, Trajectory]] = None) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Create training data from trajectories in the format expected by SIAT model.
        
        This extracts overlapping windows from trajectories to create:
        - observations: (num_samples, obs_len, 2)
        - futures: (num_samples, pred_len, 2)
        - windows: (num_samples,) list of scene contexts
        
        Args:
            obs_len: Number of observation timesteps
            pred_len: Number of prediction timesteps
            trajectories: Trajectories to use (None = all trajectories)
            
        Returns:
            Tuple of (observations, futures, windows)
        """
        if trajectories is None:
            trajectories = self.get_trajectories()
        
        if len(trajectories) == 0:
            logger.warning("No valid trajectories found")
            return np.array([]), np.array([]), []
        
        observations = []
        futures = []
        windows = []
        
        # Convert trajectories dict to list for easier indexing
        traj_list = list(trajectories.values())
        
        # For each trajectory, create observation/prediction pairs
        for target_idx, target_traj in enumerate(traj_list):
            # Create a time index for this trajectory
            frame_to_pos = {f: pos for f, pos in zip(target_traj.frames, target_traj.positions)}
            
            # Slide window through trajectory
            for i in range(len(target_traj.positions) - obs_len - pred_len + 1):
                # Get observation and future for target
                obs = target_traj.positions[i:i + obs_len].copy()
                fut = target_traj.positions[i + obs_len:i + obs_len + pred_len].copy()
                
                # Build scene window (all agents in the scene at this time)
                frames_in_window = set(target_traj.frames[i:i + obs_len + pred_len])
                
                # Collect all agents in this time window
                agents_in_window = []
                
                # Add target agent first (index 0)
                agents_in_window.append(target_traj.positions[i:i + obs_len + pred_len].copy())
                
                # Add other agents
                for other_idx, other_traj in enumerate(traj_list):
                    if other_idx == target_idx:
                        continue
                    
                    # Check if other agent overlaps with this window
                    other_frames = set(other_traj.frames[i:i + obs_len + pred_len]) & frames_in_window
                    if len(other_frames) > 0:
                        # Extract this agent's positions for the window frames
                        window_positions = []
                        for frame in range(len(target_traj.frames[i:i + obs_len + pred_len])):
                            global_frame_idx = i + frame
                            if global_frame_idx < len(target_traj.frames):
                                frame_id = target_traj.frames[global_frame_idx]
                                # Find position in other trajectory
                                if frame_id in other_traj.frames:
                                    pos_idx = other_traj.frames.index(frame_id)
                                    window_positions.append(other_traj.positions[pos_idx])
                                else:
                                    # Use last known position
                                    if len(window_positions) > 0:
                                        window_positions.append(window_positions[-1])
                        
                        if len(window_positions) == obs_len + pred_len:
                            agents_in_window.append(np.array(window_positions))
                
                # Convert to numpy array and pad if necessary
                if len(agents_in_window) > 0:
                    # Pad to same size
                    max_agents = len(agents_in_window)
                    window_array = np.zeros((max_agents, obs_len + pred_len, 2))
                    for agent_idx, agent_traj in enumerate(agents_in_window):
                        if len(agent_traj) == obs_len + pred_len:
                            window_array[agent_idx] = agent_traj
                    
                    observations.append(obs)
                    futures.append(fut)
                    windows.append(window_array)
        
        logger.info(f"Created {len(observations)} training samples from {len(trajectories)} trajectories")
        
        return (np.array(observations) if observations else np.array([]),
                np.array(futures) if futures else np.array([]),
                windows)


def extract_trajectories_from_detections(detections: Dict[int, np.ndarray],
                                         max_disappeared: int = 50,
                                         max_distance: float = 50,
                                         min_trajectory_length: int = 2) -> Dict[int, Trajectory]:
    """
    Convenience function to extract trajectories from detection dictionary.
    
    Args:
        detections: Dictionary from VideoProcessor with {frame_id: detections_array}
        max_disappeared: Max frames a person can be missing
        max_distance: Max distance for matching detections
        min_trajectory_length: Minimum trajectory length to keep
        
    Returns:
        Dictionary of extracted trajectories
    """
    extractor = TrajectoryExtractor(max_disappeared=max_disappeared,
                                   max_distance=max_distance,
                                   min_trajectory_length=min_trajectory_length)
    
    # Process frames in order
    sorted_frames = sorted(detections.keys())
    for frame_id in sorted_frames:
        dets = detections[frame_id]
        if len(dets) > 0:
            extractor.process_frame(frame_id, dets)
    
    return extractor.get_trajectories()
