"""
Video processing module for extracting pedestrian detections from video files.
Uses YOLOv8 for lightweight, fast object detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from ultralytics import YOLO
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Process video files and extract pedestrian bounding boxes using YOLOv8.
    
    Attributes:
        model: YOLOv8 model for object detection
        confidence_threshold: Confidence threshold for detections
        device: Device to run inference on ('cpu' or 'cuda')
    """
    
    PERSON_CLASS_ID = 0  # COCO class 0 is 'person'
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float = 0.5, device: str = 'cuda'):
        """
        Initialize VideoProcessor with YOLOv8 model.
        
        Args:
            model_name: Name of YOLOv8 model ('yolov8n', 'yolov8s', 'yolov8m', etc.)
                       'n' = nano (fastest, least accurate)
                       's' = small
                       'm' = medium
                       'l' = large
            confidence_threshold: Minimum confidence score for detections (0-1)
            device: Device to run on ('cpu' or 'cuda')
        """
        logger.info(f"Loading YOLOv8 model: {model_name} on device: {device}")
        self.model = YOLO(model_name)
        self.model.to(device)
        self.confidence_threshold = confidence_threshold
        self.device = device
        logger.info(f"Model loaded successfully. Confidence threshold: {confidence_threshold}")
    
    def process_video(self, video_path: str, sample_rate: int = 1) -> Dict[int, np.ndarray]:
        """
        Process a video file and extract pedestrian detections.
        
        Args:
            video_path: Path to video file (.avi, .mp4, etc.)
            sample_rate: Process every nth frame (1 = all frames, 2 = every other frame, etc.)
            
        Returns:
            Dictionary with frame indices as keys and detected bounding boxes as values.
            Format: {frame_id: [[x1, y1, x2, y2, conf, person_id], ...], ...}
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {frame_width}x{frame_height} @ {fps} fps, {total_frames} frames")
        
        detections = {}
        frame_count = 0
        processed_frames = 0
        
        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame based on sample_rate
            if frame_count % sample_rate == 0:
                # Run YOLO inference
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Extract person detections
                frame_detections = self._extract_person_detections(results[0])
                
                if len(frame_detections) > 0:
                    detections[frame_count] = frame_detections
                
                processed_frames += 1
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        logger.info(f"Video processing complete. Processed {processed_frames} frames, "
                   f"found detections in {len(detections)} frames")
        
        return detections
    
    def _extract_person_detections(self, results) -> np.ndarray:
        """
        Extract person class detections from YOLO results.
        
        Args:
            results: YOLO detection results object
            
        Returns:
            Array of shape (N, 6) with [x1, y1, x2, y2, conf, class_id]
        """
        if results.boxes is None or len(results.boxes) == 0:
            return np.array([])
        
        # Get boxes, confidences, and class IDs
        boxes = results.boxes.xyxy.cpu().numpy()  # (N, 4) - [x1, y1, x2, y2]
        confs = results.boxes.conf.cpu().numpy()   # (N,)
        class_ids = results.boxes.cls.cpu().numpy() # (N,)
        
        # Filter for person class only
        person_mask = class_ids == self.PERSON_CLASS_ID
        person_boxes = boxes[person_mask]
        person_confs = confs[person_mask]
        
        if len(person_boxes) == 0:
            return np.array([])
        
        # Combine into single array
        detections = np.hstack([
            person_boxes,
            person_confs.reshape(-1, 1)
        ])
        
        return detections
    
    def process_videos_batch(self, video_dir: str, pattern: str = '*.avi', 
                            sample_rate: int = 1) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Process multiple videos from a directory.
        
        Args:
            video_dir: Directory containing video files
            pattern: File pattern to match (e.g., '*.avi', '*.mp4')
            sample_rate: Process every nth frame
            
        Returns:
            Dictionary mapping video names to their detections
        """
        video_dir = Path(video_dir)
        video_files = sorted(video_dir.glob(pattern))
        
        logger.info(f"Found {len(video_files)} video files in {video_dir}")
        
        all_detections = {}
        for video_file in video_files:
            logger.info(f"\n--- Processing {video_file.name} ---")
            try:
                detections = self.process_video(str(video_file), sample_rate=sample_rate)
                all_detections[video_file.name] = detections
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {e}")
                continue
        
        return all_detections


def visualize_detections(video_path: str, detections: Dict[int, np.ndarray], 
                        output_path: str = None, skip_frames: int = 1):
    """
    Visualize detections on video and optionally save to file.
    
    Args:
        video_path: Path to input video
        detections: Detection results from VideoProcessor
        output_path: Path to save visualization video (None = display only)
        skip_frames: Show every nth frame (for faster playback)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw detections if available for this frame
        if frame_count in detections:
            frame = _draw_bboxes(frame, detections[frame_count])
        
        if output_path:
            out.write(frame)
        else:
            cv2.imshow('Detections', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


def _draw_bboxes(frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes on frame.
    
    Args:
        frame: Input frame
        detections: Array of [x1, y1, x2, y2, conf] detections
        
    Returns:
        Frame with drawn bounding boxes
    """
    frame = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2, conf = det[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence score
        label = f"person {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame
