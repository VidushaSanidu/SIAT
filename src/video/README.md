# Video Processing Module Documentation

## Overview

The `src/video/` module provides a complete pipeline to extract pedestrian trajectory data from video files using state-of-the-art object detection (YOLOv8) and centroid-based tracking.

## Module Structure

```
src/video/
├── __init__.py              # Module interface
├── video_processor.py       # YOLO detection
└── trajectory_extractor.py  # Tracking & data conversion
```

## Key Classes

### VideoProcessor

**Purpose**: Detect pedestrians in video frames using YOLOv8.

**Constructor**:
```python
VideoProcessor(
    model_name: str = 'yolov8n.pt',
    confidence_threshold: float = 0.5,
    device: str = 'cuda'
)
```

**Parameters**:
- `model_name`: YOLOv8 model size
  - `'yolov8n.pt'`: Nano (fastest) ~7 MB
  - `'yolov8s.pt'`: Small ~20 MB
  - `'yolov8m.pt'`: Medium ~50 MB
  - `'yolov8l.pt'`: Large (most accurate) ~100 MB
- `confidence_threshold`: Detection confidence 0-1 (higher = fewer detections)
- `device`: `'cuda'` (GPU) or `'cpu'`

**Main Method**:
```python
def process_video(self, video_path: str, sample_rate: int = 1) -> Dict[int, np.ndarray]:
    """
    Process video and return detections per frame.
    
    Args:
        video_path: Path to video file
        sample_rate: Process every nth frame (1=all, 2=every other)
    
    Returns:
        {frame_id: [[x1, y1, x2, y2, conf], ...], ...}
    """
```

**Example**:
```python
processor = VideoProcessor(model_name='yolov8n.pt', device='cuda')
detections = processor.process_video('pedestrian_video.avi', sample_rate=1)
# Returns: {0: array([[100, 150, 150, 300, 0.95], ...]), 1: array([...]), ...}
```

---

### CentroidTracker

**Purpose**: Track pedestrians across frames using centroid-based matching.

**Algorithm**: 
1. Compute centroid of each bounding box
2. Match centroids between consecutive frames
3. Assign/update/create tracks based on distance

**Constructor**:
```python
CentroidTracker(
    max_disappeared: int = 50,
    max_distance: float = 50
)
```

**Parameters**:
- `max_disappeared`: Frames to wait before removing a track (handles occlusions)
- `max_distance`: Max distance (pixels) for matching centroids

**Main Method**:
```python
def update(self, detections: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Update tracker with new frame detections.
    
    Args:
        detections: [[x1, y1, x2, y2, ...], ...] shape (N, 4+)
    
    Returns:
        {object_id: [centroid_x, centroid_y], ...}
    """
```

**Example**:
```python
tracker = CentroidTracker(max_disappeared=50, max_distance=50)

for frame_id, frame_detections in detections.items():
    objects = tracker.update(frame_detections)
    # objects = {0: [125, 225], 1: [375, 200], ...}
```

---

### Trajectory

**Purpose**: Container for a pedestrian's path through the video.

**Attributes**:
```python
@dataclass
class Trajectory:
    pedestrian_id: int          # Unique ID
    frames: List[int]           # Frame numbers where pedestrian appears
    positions: np.ndarray       # Shape (N, 2) - x,y coordinates
```

**Methods**:
```python
@property
def length(self) -> int:
    """Number of frames in trajectory."""

def get_segment(self, start_frame: int, end_frame: int) -> Optional[np.ndarray]:
    """Extract trajectory between two frames."""
```

**Example**:
```python
trajectory = Trajectory(
    pedestrian_id=0,
    frames=[0, 1, 2, ..., 280],
    positions=np.array([[100, 150], [102, 155], ...])
)

print(f"Pedestrian {trajectory.pedestrian_id} appears for {trajectory.length} frames")
segment = trajectory.get_segment(start_frame=10, end_frame=20)
```

---

### TrajectoryExtractor

**Purpose**: Extract continuous trajectories and create SIAT-compatible training data.

**Constructor**:
```python
TrajectoryExtractor(
    max_disappeared: int = 50,
    max_distance: float = 50,
    min_trajectory_length: int = 2
)
```

**Main Methods**:
```python
def process_frame(self, frame_id: int, detections: np.ndarray) -> None:
    """Process a single frame of detections."""

def get_trajectories(self) -> Dict[int, Trajectory]:
    """Get all extracted trajectories (filtered by min length)."""

def create_training_data(
    self,
    obs_len: int = 8,
    pred_len: int = 12,
    trajectories: Optional[Dict[int, Trajectory]] = None
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Create training data from trajectories.
    
    Returns:
        (observations, futures, windows) tuples
    """
```

**Example**:
```python
extractor = TrajectoryExtractor(max_disappeared=50, max_distance=50)

# Process frames sequentially
for frame_id in sorted(detections.keys()):
    extractor.process_frame(frame_id, detections[frame_id])

# Get trajectories
trajectories = extractor.get_trajectories()
print(f"Extracted {len(trajectories)} trajectories")

# Create training data
obs, futures, windows = extractor.create_training_data(obs_len=8, pred_len=12)
print(f"Training samples: {len(obs)}")  # Number of obs/future pairs
print(f"Observations shape: {obs.shape}")  # (N, 8, 2)
print(f"Futures shape: {futures.shape}")   # (N, 12, 2)
```

---

## Functions

### extract_trajectories_from_detections()

**Purpose**: Convenience function to extract trajectories from detection dict.

```python
def extract_trajectories_from_detections(
    detections: Dict[int, np.ndarray],
    max_disappeared: int = 50,
    max_distance: float = 50,
    min_trajectory_length: int = 2
) -> Dict[int, Trajectory]:
    """Convert detections to tracked trajectories."""
```

**Example**:
```python
trajectories = extract_trajectories_from_detections(
    detections,
    max_disappeared=50,
    max_distance=50,
    min_trajectory_length=8
)
```

---

### visualize_detections()

**Purpose**: Visualize detections on video.

```python
def visualize_detections(
    video_path: str,
    detections: Dict[int, np.ndarray],
    output_path: str = None,
    skip_frames: int = 1
) -> None:
```

**Example**:
```python
# Display in real-time
visualize_detections('video.avi', detections, output_path=None)

# Save to file
visualize_detections(
    'video.avi',
    detections,
    output_path='output_with_boxes.mp4'
)
```

---

## Complete Usage Example

```python
from src.video import (
    VideoProcessor,
    extract_trajectories_from_detections,
    TrajectoryExtractor,
    visualize_detections
)
import numpy as np

# Step 1: Detect pedestrians
print("Step 1: Detecting pedestrians...")
processor = VideoProcessor(model_name='yolov8n.pt', device='cuda')
detections = processor.process_video('video.avi', sample_rate=1)
print(f"Found detections in {len(detections)} frames")

# Step 2: Extract trajectories
print("\nStep 2: Extracting trajectories...")
trajectories = extract_trajectories_from_detections(detections)
print(f"Extracted {len(trajectories)} trajectories")

# Step 3: Create training data
print("\nStep 3: Creating training data...")
extractor = TrajectoryExtractor()
extractor.trajectories = trajectories
obs, futures, windows = extractor.create_training_data(obs_len=8, pred_len=12)
print(f"Created {len(obs)} training samples")

# Step 4: Save to NPZ
print("\nStep 4: Saving to NPZ...")
np.savez_compressed(
    'trajectory_data.npz',
    observations=obs,
    futures=futures,
    windows=np.array(windows, dtype=object)
)

# Step 5: Visualize (optional)
print("\nStep 5: Creating visualization...")
visualize_detections('video.avi', detections, output_path='video_with_detections.mp4')

print("\nDone! Ready to use with SIAT model.")
```

---

## Data Formats

### Input: Detection Dictionary

```python
detections = {
    frame_id: np.ndarray([
        [x1, y1, x2, y2, confidence],
        [x1, y1, x2, y2, confidence],
        ...
    ]),
    ...
}
```

**Shape**: Variable (N_people, 5) per frame
**Type**: float32
**Content**: Bounding boxes in [x1, y1, x2, y2, conf] format

---

### Output: Training Data

#### observations
**Shape**: (num_samples, 8, 2)
**Type**: float32
**Content**: Target pedestrian's observed trajectory (8 timesteps, x,y coords)

#### futures
**Shape**: (num_samples, 12, 2)
**Type**: float32
**Content**: Target pedestrian's future trajectory (12 timesteps, x,y coords)

#### windows
**Type**: List of np.ndarray
**Shape per item**: (n_agents, 20, 2) where 20 = obs_len + pred_len
**Content**: All agents in the scene context for each sample

---

## Configuration Parameters

### YOLOv8 Detection

| Parameter | Range | Recommendation |
|-----------|-------|-----------------|
| confidence | 0.0-1.0 | 0.4-0.6 |
| model_size | n,s,m,l | n (speed) or s (balanced) |
| sample_rate | 1+ | 1 (all frames) or 2 (every other) |

### Centroid Tracking

| Parameter | Typical | Effect |
|-----------|---------|--------|
| max_disappeared | 20-100 | Higher = more forgiving |
| max_distance | 30-100 | Higher = matches farther gaps |
| min_trajectory_length | 5-20 | Higher = longer trajectories only |

### Training Data Creation

| Parameter | Typical | Notes |
|-----------|---------|-------|
| obs_len | 8 | Standard SIAT observation length |
| pred_len | 12 | Standard SIAT prediction length |

---

## Performance Tuning

### For Speed

```python
processor = VideoProcessor(model_name='yolov8n.pt', device='cuda')
detections = processor.process_video('video.avi', sample_rate=2)
trajectories = extract_trajectories_from_detections(detections)
```

### For Accuracy

```python
processor = VideoProcessor(model_name='yolov8m.pt', device='cuda')
detections = processor.process_video('video.avi', sample_rate=1)
trajectories = extract_trajectories_from_detections(
    detections,
    max_distance=100,
    max_disappeared=100
)
```

### For Crowded Scenes

```python
processor = VideoProcessor(
    model_name='yolov8m.pt',
    confidence_threshold=0.4,
    device='cuda'
)
detections = processor.process_video('video.avi', sample_rate=1)
trajectories = extract_trajectories_from_detections(
    detections,
    max_distance=100
)
```

---

## Error Handling

```python
try:
    detections = processor.process_video('nonexistent.avi')
except FileNotFoundError:
    print("Video file not found")
except RuntimeError as e:
    print(f"Failed to process video: {e}")
```

---

## Integration with SIAT

### Direct Integration

```python
from src.video import extract_trajectories_from_detections, TrajectoryExtractor
from src.data import TrajectoryDataset
from torch.utils.data import DataLoader
from src.data.dataset import collate_fn

# 1. Get trajectories from video
trajectories = extract_trajectories_from_detections(detections)

# 2. Create training data
extractor = TrajectoryExtractor()
extractor.trajectories = trajectories
obs, futures, windows = extractor.create_training_data(obs_len=8, pred_len=12)

# 3. Save to NPZ
np.savez_compressed('video_data.npz', observations=obs, futures=futures, windows=windows)

# 4. Use with DataLoader
dataset = TrajectoryDataset(['video_data.npz'])
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# 5. Train model
for batch in dataloader:
    obs, fut, window, mask = batch['obs'], batch['fut'], batch['window'], batch['agent_mask']
    predictions = model(obs, window, mask)
```

---

## Logging

All modules use Python's logging. Enable detailed output:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('src.video')
logger.setLevel(logging.DEBUG)

# Now all video processing will show detailed logs
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| No detections | Low confidence or poor video | Lower `confidence_threshold` |
| Fragmented tracks | Tracking too strict | Increase `max_distance`, `max_disappeared` |
| Out of memory | Video too large | Use `sample_rate=2` or smaller model |
| Slow processing | Large model or CPU | Use `yolov8n.pt` or `device='cuda'` |
| CUDA error | Wrong PyTorch/CUDA | Check PyTorch installation |

---

## References

- YOLO: https://github.com/ultralytics/ultralytics
- Centroid Tracking: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
- COCO Classes: https://cocodataset.org/

---

## API Summary

```python
# Main classes
VideoProcessor(model_name, confidence_threshold, device)
CentroidTracker(max_disappeared, max_distance)
Trajectory(pedestrian_id, frames, positions)
TrajectoryExtractor(max_disappeared, max_distance, min_trajectory_length)

# Main functions
extract_trajectories_from_detections(detections, **params)
visualize_detections(video_path, detections, output_path, skip_frames)
```
