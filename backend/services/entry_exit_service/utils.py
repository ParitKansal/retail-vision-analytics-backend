"""
Service-specific utilities for Entry/Exit Counting
Contains ThreeLineCounter and related classes
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.error("ultralytics not available. Please install: pip install ultralytics")

# Try to import ByteTracker from ultralytics (optional, more robust tracking)
try:
    from ultralytics.trackers import BYTETracker
    BYTETRACKER_AVAILABLE = True
except ImportError:
    BYTETRACKER_AVAILABLE = False
    logger.debug("ByteTracker not available. Using improved tracker instead.")

# Try to import DeepFace for gender/age detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not available. Gender/age detection will be disabled. Install with: pip install deepface")

# Default YOLO model path (resolved relative to this file)
DEFAULT_MODEL_PATH = str(Path(__file__).parent.resolve() / "artifacts" / "yolov8l.pt")


class YOLOWorldClient:
    """
    Client for local YOLOv8 model.

    In this service we use a standard YOLOv8-L model (COCO pre-trained) and
    filter detections to the **person** class only (class id 0).
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, confidence: float = 0.3):
        """
        Initialize YOLOv8 client.

        Args:
            model_path: Path to YOLOv8 model file (e.g. yolov8l.pt)
            confidence: Detection confidence threshold
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Please install: pip install ultralytics")

        # Resolve model path
        model_path = Path(model_path)
        if not model_path.is_absolute():
            current_dir = Path(__file__).parent
            model_path = current_dir / model_path

        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")

        logger.info(f"Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

        logger.info("YOLOv8 model loaded successfully")

    def detect_objects(self, frame: np.ndarray, text_prompt: str) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Detect people in a frame using YOLOv8.

        Args:
            frame: Input frame in BGR format (numpy array)
            text_prompt: Ignored (kept for backward compatibility)

        Returns:
            List of (x1, y1, x2, y2, confidence, label) tuples for detected persons.
        """
        try:
            self.request_count += 1

            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                verbose=False
            )

            detections: List[Tuple[int, int, int, int, float, str]] = []
            h, w = frame.shape[:2]

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        try:
                            box = boxes.xyxy[i].cpu().numpy()
                            x1, y1, x2, y2 = map(float, box[:4])
                            conf = float(boxes.conf[i].cpu().numpy())
                            cls_id = int(boxes.cls[i].cpu().numpy())

                            # Keep only person class (0) for counting
                            if cls_id != 0:
                                continue

                            label = "person"

                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            x1 = max(0, min(w - 1, x1))
                            y1 = max(0, min(h - 1, y1))
                            x2 = max(0, min(w - 1, x2))
                            y2 = max(0, min(h - 1, y2))

                            if x2 > x1 and y2 > y1:
                                detections.append((x1, y1, x2, y2, conf, label))

                        except Exception as e:
                            logger.warning(f"Error processing detection {i}: {e}")
                            continue

            if detections:
                self.success_count += 1

            return detections

        except Exception as e:
            logger.error(f"Error in detect_objects: {e}", exc_info=True)
            self.error_count += 1
            return []
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            "total_requests": self.request_count,
            "successful": self.success_count,
            "errors": self.error_count,
            "success_rate": (self.success_count / self.request_count * 100) if self.request_count > 0 else 0
        }


class GenderAgeDetector:
    """Gender and age detection using DeepFace"""
    
    def __init__(self):
        """Initialize gender and age detector using DeepFace"""
        self.model_available = False
        
        if DEEPFACE_AVAILABLE:
            try:
                self.DeepFace = DeepFace
                self.model_available = True
                logger.info("DeepFace model loaded for gender/age detection")
            except Exception as e:
                logger.warning(f"DeepFace initialization failed: {e}")
                self.model_available = False
        else:
            logger.warning("DeepFace not available. Gender/age detection disabled.")
    
    def detect_gender_age(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect gender and age group from person bounding box.
        
        Args:
            frame: Full frame in BGR format
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Tuple of (gender, age_group) where:
            - gender: 'male', 'female', or None
            - age_group: 'child', 'young_adult', 'middle_aged', 'senior', or None
        """
        if not self.model_available:
            return None, None
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return None, None
        
        # Extract face region (upper 40% of person bounding box)
        face_height = int((y2 - y1) * 0.4)
        face_roi = person_roi[:face_height, :]
        
        if face_roi.size == 0:
            face_roi = person_roi
        
        try:
            if hasattr(self, 'DeepFace'):
                return self._detect_with_deepface(face_roi)
        except Exception as e:
            logger.debug(f"Error in gender/age detection: {e}")
            return None, None
        
        return None, None
    
    def _detect_with_deepface(self, face_roi: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        """Detect gender and age using DeepFace"""
        try:
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            result = self.DeepFace.analyze(
                face_rgb,
                actions=['gender', 'age'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            gender = result.get('dominant_gender', '').lower()
            if 'man' in gender or 'male' in gender:
                gender = 'male'
            elif 'woman' in gender or 'female' in gender:
                gender = 'female'
            else:
                gender = None
            
            age = result.get('age', None)
            age_group = self._categorize_age(age) if age else None
            
            return gender, age_group
            
        except Exception as e:
            logger.debug(f"DeepFace detection failed: {e}")
            return None, None
    
    def _categorize_age(self, age: float) -> Optional[str]:
        """Categorize age into groups"""
        if age is None:
            return None
        
        if age < 18:
            return 'child'
        elif age < 35:
            return 'young_adult'
        elif age < 60:
            return 'middle_aged'
        else:
            return 'senior'


class ImprovedTracker:
    """
    Improved object tracker with velocity prediction and longer track persistence.
    Maintains tracks even when detection is temporarily lost.
    """
    
    def __init__(self, max_distance: float = 150.0, max_frames_lost: int = 15):
        """
        Initialize improved tracker.
        
        Args:
            max_distance: Maximum distance for track association (increased for better matching)
            max_frames_lost: Maximum frames to keep lost tracks (increased to handle occlusions)
        """
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.next_track_id = 0
        self.tracked_objects: Dict[int, Dict] = {}
    
    def _predict_position(self, track_info: Dict) -> Tuple[float, float]:
        """
        Predict next position based on velocity.
        
        Args:
            track_info: Track information dictionary
            
        Returns:
            Predicted (x, y) position
        """
        if 'velocity' not in track_info or track_info['velocity'] is None:
            # No velocity info, return last known position
            return track_info['centroid']
        
        vx, vy = track_info['velocity']
        last_cx, last_cy = track_info['centroid']
        
        # Predict position based on velocity
        predicted_x = last_cx + vx
        predicted_y = last_cy + vy
        
        return (predicted_x, predicted_y)
    
    def _calculate_velocity(self, track_info: Dict, new_centroid: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate velocity from position history.
        
        Args:
            track_info: Track information dictionary
            new_centroid: New centroid position
            
        Returns:
            (vx, vy) velocity vector
        """
        if 'centroid_history' not in track_info or len(track_info['centroid_history']) < 2:
            return (0.0, 0.0)
        
        history = track_info['centroid_history']
        if len(history) >= 2:
            # Use last 2 positions to calculate velocity
            prev_cx, prev_cy = history[-1]
            prev_prev_cx, prev_prev_cy = history[-2] if len(history) >= 2 else history[-1]
            
            # Average velocity over last 2 frames
            vx1 = prev_cx - prev_prev_cx
            vy1 = prev_cy - prev_prev_cy
            vx2 = new_centroid[0] - prev_cx
            vy2 = new_centroid[1] - prev_cy
            
            # Exponential moving average for smoother velocity
            alpha = 0.7
            vx = alpha * vx2 + (1 - alpha) * vx1
            vy = alpha * vy2 + (1 - alpha) * vy1
            
            return (vx, vy)
        
        return (0.0, 0.0)
    
    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Update tracks with new detections, using velocity prediction for lost tracks.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) detections
            
        Returns:
            List of (x1, y1, x2, y2, confidence, track_id) tracked detections
        """
        tracked = []
        used_tracks = set()
        
        # First, predict positions for all existing tracks
        predicted_positions = {}
        for track_id, track_info in self.tracked_objects.items():
            if track_info.get('last_seen', 0) > 0:
                # Track is lost, use predicted position
                predicted_positions[track_id] = self._predict_position(track_info)
            else:
                # Track is active, use current position
                predicted_positions[track_id] = track_info['centroid']
        
        # Match detections to tracks
        for det in detections:
            x1, y1, x2, y2, conf = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            new_centroid = (cx, cy)
            
            best_track_id = None
            best_distance = self.max_distance
            
            # Try to match with existing tracks (including lost ones)
            for track_id, track_info in self.tracked_objects.items():
                if track_id in used_tracks:
                    continue
                
                # Use predicted position for lost tracks, actual position for active tracks
                if track_id in predicted_positions:
                    pred_cx, pred_cy = predicted_positions[track_id]
                else:
                    pred_cx, pred_cy = track_info['centroid']
                
                # Calculate distance to predicted/current position
                distance = np.sqrt((cx - pred_cx)**2 + (cy - pred_cy)**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                track_id = best_track_id
                used_tracks.add(track_id)
                track_info = self.tracked_objects[track_id]
                
                # Update centroid and calculate velocity
                old_centroid = track_info['centroid']
                track_info['centroid'] = new_centroid
                track_info['bbox'] = (x1, y1, x2, y2)
                track_info['last_seen'] = 0  # Reset - track is now active
                
                # Update centroid history
                if 'centroid_history' not in track_info:
                    track_info['centroid_history'] = deque(maxlen=10)
                track_info['centroid_history'].append(new_centroid)
                
                # Calculate and update velocity
                velocity = self._calculate_velocity(track_info, new_centroid)
                track_info['velocity'] = velocity
                
                tracked.append((x1, y1, x2, y2, conf, track_id))
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracked_objects[track_id] = {
                    'centroid': new_centroid,
                    'bbox': (x1, y1, x2, y2),
                    'last_seen': 0,
                    'velocity': (0.0, 0.0),
                    'centroid_history': deque([new_centroid], maxlen=10)
                }
                tracked.append((x1, y1, x2, y2, conf, track_id))
        
        # Update lost tracks and predict their positions
        for track_id, track_info in self.tracked_objects.items():
            if track_id not in used_tracks:
                track_info['last_seen'] = track_info.get('last_seen', 0) + 1
                
                # If track is lost but not too old, predict position and add to tracked list
                if track_info['last_seen'] <= self.max_frames_lost:
                    # Predict position based on velocity
                    pred_cx, pred_cy = self._predict_position(track_info)
                    
                    # Update predicted centroid
                    track_info['centroid'] = (pred_cx, pred_cy)
                    
                    # Get last known bbox and expand slightly
                    last_bbox = track_info.get('bbox', (0, 0, 0, 0))
                    if last_bbox != (0, 0, 0, 0):
                        x1, y1, x2, y2 = last_bbox
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Create predicted bbox around predicted centroid
                        pred_x1 = int(pred_cx - width / 2)
                        pred_y1 = int(pred_cy - height / 2)
                        pred_x2 = int(pred_cx + width / 2)
                        pred_y2 = int(pred_cy + height / 2)
                        
                        # Add predicted track with lower confidence
                        tracked.append((pred_x1, pred_y1, pred_x2, pred_y2, 0.3, track_id))
        
        # Remove very old tracks
        tracks_to_remove = [
            tid for tid, info in self.tracked_objects.items()
            if info.get('last_seen', 0) > self.max_frames_lost
        ]
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
        
        return tracked


# Keep SimpleTracker as fallback
SimpleTracker = ImprovedTracker


@dataclass
class PersonInfo:
    """Information about a tracked person"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]
    trajectory: deque  # History of positions
    last_seen: int = 0
    entered: bool = False
    exited: bool = False
    label: str = "people"
    entry_time: Optional[float] = None
    exit_time: Optional[float] = None
    is_in_group: bool = False
    gender: Optional[str] = None
    age_group: Optional[str] = None
    # Track which lines have been crossed
    lines_crossed: deque = None  # Will store line numbers (1, 2, 3) in order
    last_side: Optional[str] = None  # 'left' or 'right' - which side of line 2 they're on
    line_sides: Dict = None  # Track which side of each line the person is on


class ThreeLineCounter:
    """Entrance counter system using three-line crossing detection"""
    
    def __init__(
        self,
        camera_id: int = 2,
        lines: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,  # List of 3 lines, each line is ((x1,y1), (x2,y2))
        model_path: str = DEFAULT_MODEL_PATH,
        text_prompt: str = "people",
        frame_skip: int = 3,
        confidence: float = 0.3,
        enable_gender_age: bool = True,
        use_bytetracker: bool = False,
        filter_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,  # Filter line to ignore detections on left side
        filter_line_offset: Optional[float] = None  # Offset distance in pixels to move filter line left (None = auto-calculate)
    ):
        """
        Initialize three-line entrance counter.
        
        Args:
            camera_id: Camera ID (default: 2)
            lines: List of 3 lines, each line is ((x1,y1), (x2,y2))
                   Line 1 = left, Line 2 = middle, Line 3 = right
            model_path: Path to YOLOv8 model file (e.g. yolov8l.pt)
            text_prompt: Text prompt for object detection
            frame_skip: Process every Nth frame
            confidence: Detection confidence threshold
            enable_gender_age: Enable gender and age detection
            use_bytetracker: If True, use ByteTracker from ultralytics (more robust but requires ultralytics)
            filter_line: Optional filter line ((x1,y1), (x2,y2)) - detections on left side will be ignored
            filter_line_offset: Offset distance in pixels to move filter line left from line_1 (None = auto-calculate)
        """
        self.camera_id = camera_id
        self.lines = lines if lines else []  # List of 3 lines
        self.filter_line = filter_line  # Filter line to ignore detections on left side
        self.filter_line_offset = filter_line_offset  # Offset distance for filter line
        self.frame_skip = frame_skip
        self.confidence = confidence
        self.text_prompt = text_prompt.strip()
        self.use_bytetracker = use_bytetracker and BYTETRACKER_AVAILABLE
        
        # Circular (radial) zone configuration for entry/exit detection.
        # When enabled, three concentric circular boundaries (like three half-circles)
        # are used instead of three straight lines.
        self.use_circular_zones: bool = False
        self.circle_center: Optional[Tuple[float, float]] = None  # (cx, cy)
        self.circle_radii: Optional[List[float]] = None           # [r1, r2, r3]
        
        # Initialize components
        self.yolo_client = YOLOWorldClient(model_path=model_path, confidence=confidence)
        
        # Choose tracker
        if self.use_bytetracker:
            # ByteTracker is more robust for tracking through occlusions
            if BYTETRACKER_AVAILABLE:
                # Initialize ByteTracker with appropriate parameters
                self.tracker = BYTETracker(
                    track_thresh=0.25,  # Detection threshold
                    track_buffer=30,    # Number of frames to buffer for lost tracks
                    match_thresh=0.8,   # Matching threshold
                    frame_rate=30       # Frame rate (will be updated per frame)
                )
                logger.info("Using ByteTracker for improved tracking robustness")
            else:
                logger.warning("ByteTracker requested but not available. Falling back to ImprovedTracker.")
                self.tracker = ImprovedTracker(max_distance=150.0, max_frames_lost=15)
                self.use_bytetracker = False
        else:
            # Use improved tracker with longer persistence and velocity prediction
            self.tracker = ImprovedTracker(max_distance=150.0, max_frames_lost=15)
            logger.info("Using ImprovedTracker with velocity prediction")
        
        # Initialize gender/age detector using DeepFace
        self.gender_age_detector = GenderAgeDetector() if enable_gender_age else None
        
        # Tracking
        self.tracked_people: Dict[int, PersonInfo] = {}
        self.frame_count = 0
        self.last_raw_detections = []
        self.track_id_to_label: Dict[int, str] = {}
        
        # Counters
        self.entered_count = 0
        self.exited_count = 0
        self.individual_entered = 0
        self.group_entered = 0
        self.individual_exited = 0
        self.group_exited = 0
        
        # Gender and age statistics
        self.gender_stats = {'male': 0, 'female': 0, 'unknown': 0}
        self.age_stats = {'child': 0, 'young_adult': 0, 'middle_aged': 0, 'senior': 0, 'unknown': 0}
        
        # Group detection
        self.group_time_window = 3.0  # seconds
        self.recent_events = {
            'entered': [],  # (track_id, timestamp)
            'exited': []
        }
        
        logger.info(f"Initialized Three-Line Counter with YOLOv8 for Camera {camera_id}")
        logger.info(f"Model: {model_path}, Text Prompt: {text_prompt}, Confidence: {confidence}")
    
    def set_lines(self, lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
        """Set the three counting lines"""
        if len(lines) != 3:
            raise ValueError("Must provide exactly 3 lines")
        self.lines = lines
        logger.info(f"Three lines set: Line 1 (left), Line 2 (middle), Line 3 (right)")
    
    def set_circular_zones(self, center: Tuple[float, float], radii: List[float]):
        """
        Enable circular (radial) counting zones instead of straight lines.
        
        Args:
            center: (cx, cy) center point (typically mid of entrance door)
            radii: List of three radii [r1, r2, r3] with r1 < r2 < r3
        """
        if len(radii) != 3:
            raise ValueError("Must provide exactly 3 radii for circular zones")
        if not (radii[0] < radii[1] < radii[2]):
            raise ValueError("Radii must be strictly increasing: r1 < r2 < r3")
        
        self.circle_center = center
        self.circle_radii = radii
        self.use_circular_zones = True
        logger.info(f"Circular zones enabled with center={center}, radii={radii}")
    
    def set_text_prompt(self, text_prompt: str):
        """Update text prompt for detection"""
        self.text_prompt = text_prompt.strip()
        logger.info(f"Text prompt updated: {text_prompt}")
    
    def _point_to_line_side(self, point: Tuple[float, float], line: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
        """
        Determine which side of a line a point is on.
        In image coordinates: (0,0) is top-left, x increases right, y increases down.
        "Right" means the point is to the right of the line (higher x-coordinate in image).
        
        Args:
            point: (x, y) point
            line: ((x1, y1), (x2, y2)) line endpoints
            
        Returns:
            'left' or 'right' - 'right' means point is to the right of the line in image coordinates
        """
        (x1, y1), (x2, y2) = line
        px, py = point
        
        # For horizontal lines, use x-coordinate comparison directly
        if abs(y2 - y1) < 5:  # Nearly horizontal line
            return 'right' if px > min(x1, x2) else 'left'
        
        # For vertical lines, "right" means higher x-coordinate
        if abs(x2 - x1) < 5:  # Nearly vertical line
            return 'right' if px > x1 else 'left'
        
        # For diagonal lines, use cross product
        # Cross product: (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        
        # The interpretation depends on line direction
        # For a line going from (x1,y1) to (x2,y2):
        # - If line goes generally left-to-right (x2 > x1), and point is to the right,
        #   the cross product will be negative (point is "below" the line in vector math sense)
        # - But in image coordinates, we want "right" to mean higher x
        # 
        # Actually, let's use a simpler approach: find the x-coordinate of the line at point's y-level
        # and compare point's x to that
        if abs(y2 - y1) > 1e-6:
            # Line equation: x = x1 + (x2 - x1) * (py - y1) / (y2 - y1)
            line_x_at_point_y = x1 + (x2 - x1) * (py - y1) / (y2 - y1)
            return 'right' if px > line_x_at_point_y else 'left'
        else:
            # Fallback to cross product
            return 'right' if cross_product < 0 else 'left'
    
    def _calculate_filter_line(self, frame_shape: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Calculate a filter line parallel to line_1 that spans from edge to edge of the image.
        This line is positioned before (to the left of) line_1.
        
        Args:
            frame_shape: (height, width) of the frame
            
        Returns:
            Filter line ((x1, y1), (x2, y2)) or None if line_1 is not defined
        """
        if not self.lines or len(self.lines) < 1:
            return None
        
        line_1 = self.lines[0]  # Leftmost line
        (x1, y1), (x2, y2) = line_1
        h, w = frame_shape[:2]
        
        # Calculate line direction vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate line length
        line_length = np.sqrt(dx**2 + dy**2)
        if line_length < 1:
            return None
        
        # Calculate offset distance to position line to the left of line_1
        # Use configured offset or calculate based on line position
        if self.filter_line_offset is not None:
            # Use configured offset
            offset_distance = self.filter_line_offset
        else:
            # Auto-calculate: Use a reasonable offset (e.g., 50-100 pixels) or calculate based on line position
            # Find the minimum x-coordinate of line_1 to determine offset
            min_x_line1 = min(x1, x2)
            offset_distance = max(50, min_x_line1 * 0.1)  # At least 50 pixels, or 10% of min_x
        
        # Calculate perpendicular vector to move line LEFT (decrease x-coordinate)
        # For a line with direction (dx, dy) = (x2-x1, y2-y1):
        #   - Perpendicular vectors are: (-dy, dx) and (dy, -dx)
        #   - (-dy, dx) moves RIGHT (positive x when dx > 0)
        #   - (dy, -dx) moves LEFT (negative x when dx > 0)
        # We use (dy, -dx) to move LEFT
        perp_length = np.sqrt(dx**2 + dy**2)
        if perp_length < 1e-6:
            return None
        # Use (dy, -dx) normalized to move left
        perp_dx = dy / perp_length
        perp_dy = -dx / perp_length
        
        # Calculate offset - positive offset_distance moves line LEFT (decreases x)
        offset_x = perp_dx * offset_distance
        offset_y = perp_dy * offset_distance
        
        # Calculate line equation for the offset line: ax + by + c = 0
        # Original line: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        # Offset line: same coefficients but shifted
        a = dy
        b = -dx
        c_original = dx * y1 - dy * x1
        
        # Calculate c for offset line (shift by offset_distance perpendicularly)
        # For a point (x0, y0) on the original line, the offset line passes through (x0 + offset_x, y0 + offset_y)
        # Using midpoint of line_1
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        offset_mid_x = mid_x + offset_x
        offset_mid_y = mid_y + offset_y
        c_offset = -a * offset_mid_x - b * offset_mid_y
        
        # Find intersections with image boundaries
        intersections = []
        
        # Top edge (y = 0)
        if abs(b) > 1e-6:
            x_top = -c_offset / a if abs(a) > 1e-6 else None
            if x_top is not None and 0 <= x_top <= w - 1:
                intersections.append((int(x_top), 0))
        
        # Bottom edge (y = h - 1)
        if abs(b) > 1e-6:
            x_bottom = (-c_offset - b * (h - 1)) / a if abs(a) > 1e-6 else None
            if x_bottom is not None and 0 <= x_bottom <= w - 1:
                intersections.append((int(x_bottom), h - 1))
        
        # Left edge (x = 0)
        if abs(a) > 1e-6:
            y_left = -c_offset / b if abs(b) > 1e-6 else None
            if y_left is not None and 0 <= y_left <= h - 1:
                intersections.append((0, int(y_left)))
        
        # Right edge (x = w - 1)
        if abs(a) > 1e-6:
            y_right = (-c_offset - a * (w - 1)) / b if abs(b) > 1e-6 else None
            if y_right is not None and 0 <= y_right <= h - 1:
                intersections.append((w - 1, int(y_right)))
        
        # Remove duplicate intersections
        unique_intersections = []
        for inter in intersections:
            if not any(abs(inter[0] - u[0]) < 2 and abs(inter[1] - u[1]) < 2 for u in unique_intersections):
                unique_intersections.append(inter)
        
        if len(unique_intersections) >= 2:
            # Use the two intersections that are furthest apart
            max_dist = 0
            best_pair = (unique_intersections[0], unique_intersections[1])
            for i in range(len(unique_intersections)):
                for j in range(i + 1, len(unique_intersections)):
                    dist = np.sqrt((unique_intersections[i][0] - unique_intersections[j][0])**2 + 
                                 (unique_intersections[i][1] - unique_intersections[j][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (unique_intersections[i], unique_intersections[j])
            new_x1, new_y1 = best_pair[0]
            new_x2, new_y2 = best_pair[1]
        elif len(unique_intersections) == 1:
            # Only one intersection - extend line in both directions
            inter = unique_intersections[0]
            # Normalize direction vector
            dx_norm = dx / line_length
            dy_norm = dy / line_length
            # Extend along line direction
            new_x1 = max(0, min(w - 1, int(inter[0] - dx_norm * w)))
            new_y1 = max(0, min(h - 1, int(inter[1] - dy_norm * h)))
            new_x2 = max(0, min(w - 1, int(inter[0] + dx_norm * w)))
            new_y2 = max(0, min(h - 1, int(inter[1] + dy_norm * h)))
        else:
            # Fallback: use offset from original line endpoints
            new_x1 = max(0, min(w - 1, int(x1 + offset_x)))
            new_y1 = max(0, min(h - 1, int(y1 + offset_y)))
            new_x2 = max(0, min(w - 1, int(x2 + offset_x)))
            new_y2 = max(0, min(h - 1, int(y2 + offset_y)))
        
        return ((new_x1, new_y1), (new_x2, new_y2))
    
    def _line_segments_intersect(self, seg1_start: Tuple[float, float], seg1_end: Tuple[float, float],
                                 seg2_start: Tuple[float, float], seg2_end: Tuple[float, float]) -> bool:
        """
        Check if two line segments intersect.
        Uses the orientation method to determine if segments intersect.
        
        Args:
            seg1_start: Start point of first segment (x, y)
            seg1_end: End point of first segment (x, y)
            seg2_start: Start point of second segment (x, y)
            seg2_end: End point of second segment (x, y)
            
        Returns:
            True if segments intersect
        """
        def orientation(p, q, r):
            """Returns orientation of ordered triplet (p, q, r)"""
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-9:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or counterclockwise
        
        def on_segment(p, q, r):
            """Check if point q lies on segment pr"""
            if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
                return True
            return False
        
        # Find orientations
        o1 = orientation(seg1_start, seg1_end, seg2_start)
        o2 = orientation(seg1_start, seg1_end, seg2_end)
        o3 = orientation(seg2_start, seg2_end, seg1_start)
        o4 = orientation(seg2_start, seg2_end, seg1_end)
        
        # General case: segments intersect if orientations are different
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases: collinear points
        if o1 == 0 and on_segment(seg1_start, seg2_start, seg1_end):
            return True
        if o2 == 0 and on_segment(seg1_start, seg2_end, seg1_end):
            return True
        if o3 == 0 and on_segment(seg2_start, seg1_start, seg2_end):
            return True
        if o4 == 0 and on_segment(seg2_start, seg1_end, seg2_end):
            return True
        
        return False
    
    def _has_crossed_line(self, prev_point: Tuple[float, float], current_point: Tuple[float, float], 
                          line: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """
        Check if movement from prev_point to current_point crosses the line segment.
        Only counts crossings that intersect the actual line segment between endpoints,
        not the infinite line extension.
        
        Args:
            prev_point: Previous position (x, y)
            current_point: Current position (x, y)
            line: ((x1, y1), (x2, y2)) line segment endpoints
            
        Returns:
            True if line segment was crossed
        """
        (x1, y1), (x2, y2) = line
        
        # Check if the movement segment intersects with the line segment
        return self._line_segments_intersect(
            prev_point, current_point,
            (float(x1), float(y1)), (float(x2), float(y2))
        )
    
    def _get_line_number(self, line: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        """Get line number (1, 2, or 3) from line"""
        try:
            return self.lines.index(line) + 1
        except ValueError:
            return 0
    
    def _check_entry_exit(self, person: PersonInfo, crossed_line_num: int):
        """
        Check if person has completed entry or exit sequence.
        
        MAXIMUM FLEXIBILITY logic (allows missing ANY line including border lines):
        - ENTRY: Person is moving inward (line numbers increasing)
          Examples: 1-2, 2-3, 1-3, 1-2-3 (any sequence where numbers increase)
          Key: Last crossing must be greater than a previous crossing
        - EXIT: Person is moving outward (line numbers decreasing)
          Examples: 3-2, 2-1, 3-1, 3-2-1 (any sequence where numbers decrease)
          Key: Last crossing must be less than a previous crossing
        
        This is extremely robust because:
        1. It checks the full sequence history
        2. It allows missing ANY line (including border lines)
        3. It only requires directional movement (inward for entry, outward for exit)
        
        IMPORTANT: Entry and Exit are mutually exclusive - a sequence can only be one or the other.
        """
        if person.lines_crossed is None:
            person.lines_crossed = deque(maxlen=20)  # Keep last 20 line crossings
        
        # Note: crossed_line_num is already appended in the calling code
        
        # Need at least 2 crossings to determine entry/exit
        if len(person.lines_crossed) < 2:
            logger.debug(f"Person {person.track_id} - Only {len(person.lines_crossed)} line crossings, need at least 2")
            return
        
        crossed_list = list(person.lines_crossed)
        last_crossing = crossed_list[-1]  # Most recent line crossed
        
        logger.info(f"Checking entry/exit for person {person.track_id}. Crossed lines sequence: {crossed_list}, Last: {last_crossing}")
        
        # CRITICAL: Check EXIT first, then ENTRY, to ensure they are mutually exclusive
        # If a sequence matches exit criteria, it cannot be an entry
        
        # MAXIMUM FLEXIBILITY EXIT DETECTION (with mandatory line 1 presence): 
        # Exit if person is moving outward (line numbers decreasing)
        # Examples: 3-2-1, 3-1, 2-1 (any sequence where last < previous)
        # IMPORTANT: Line 1 MUST appear somewhere in the sequence for a valid EXIT.
        if last_crossing in [1, 2]:  # Exit ends with line 1 or line 2
            # Check if there's any previous crossing that's greater than the last crossing
            # This means person moved from a higher line number to a lower one (outward movement)
            has_higher_previous = any(prev_line > last_crossing for prev_line in crossed_list[:-1])
            
            logger.info(f"Person {person.track_id} - Exit check: last_crossing={last_crossing}, has_higher_previous={has_higher_previous}, entered={person.entered}, exited={person.exited}")
            logger.info(f"  Full sequence: {crossed_list}")
            
            # Exit if: There's a previous crossing with higher line number (outward movement)
            if has_higher_previous:
                # NEW RULE: For any EXIT, line 1 MUST be present somewhere in the history
                if 1 not in crossed_list:
                    logger.debug(f"Person {person.track_id} - Exit attempt ignored (line 1 not present in sequence). Sequence: {crossed_list}")
                    return
                # Check if person has crossed line 1 before in this sequence (meaning they were outside)
                has_crossed_1_before = 1 in crossed_list[:-1]
                
                # If person hasn't entered AND has crossed line 1 before, they're coming from outside (not a valid exit)
                # This means sequence like 1-2-1 (outside -> inside -> outside) - the last 1 is not an exit
                if not person.entered and has_crossed_1_before:
                    # Find if there's a pattern like 1-X-1 where X > 1 (outside -> inside -> outside)
                    # This is not a valid exit
                    first_1_index = None
                    for i, line_num in enumerate(crossed_list[:-1]):
                        if line_num == 1:
                            first_1_index = i
                            break
                    
                    if first_1_index is not None:
                        # Check if there's a higher line between first 1 and last crossing
                        has_higher_between = any(
                            crossed_list[i] > 1 
                            for i in range(first_1_index + 1, len(crossed_list) - 1)
                        )
                        if has_higher_between:
                            logger.debug(f"Person {person.track_id} - Exit attempt ignored (crossed line 1 before, coming from outside). Sequence: {crossed_list}")
                            return
                
                # If person already exited and hasn't entered again, prevent re-exit
                if person.exited and not person.entered:
                    logger.debug(f"Person {person.track_id} - Re-exit attempt ignored (already exited, need to enter first). Sequence: {crossed_list}")
                    return
                
                # If person already exited but has entered again, allow exit (new cycle)
                if person.exited and person.entered:
                    logger.info(f"Person {person.track_id} - New exit cycle (exited before, but entered again). Resetting flags.")
                    # Reset flags for new cycle
                    person.entered = False
                    person.exited = False
                
                # Valid exit: Person has moved from higher line to lower line (outward movement)
                person.exited = True
                person.exit_time = time.time()
                self.exited_count += 1
                
                logger.info(f"Person {person.track_id} - EXIT DETECTED! Sequence: {crossed_list} (outward movement), entered={person.entered}, exited={person.exited}")
                
                # Check for group exit
                self.recent_events['exited'].append((person.track_id, person.exit_time))
                self.recent_events['exited'] = [
                    (tid, ts) for tid, ts in self.recent_events['exited']
                    if person.exit_time - ts <= self.group_time_window
                ]
                
                simultaneous_exits = [
                    tid for tid, ts in self.recent_events['exited']
                    if tid != person.track_id and abs(person.exit_time - ts) <= self.group_time_window
                ]
                
                if len(simultaneous_exits) > 0:
                    person.is_in_group = True
                    self.group_exited += 1
                    for tid in simultaneous_exits:
                        if tid in self.tracked_people:
                            self.tracked_people[tid].is_in_group = True
                    logger.info(f"Person ID {person.track_id} EXITED (GROUP) - Sequence: {crossed_list}")
                else:
                    person.is_in_group = False
                    self.individual_exited += 1
                    logger.info(f"Person ID {person.track_id} EXITED (INDIVIDUAL) - Sequence: {crossed_list}")
                
                # CRITICAL: Return immediately after exit detection - do NOT check for entry
                return
        
        # MAXIMUM FLEXIBILITY ENTRY DETECTION (with mandatory line 1 presence): 
        # Entry if person is moving inward (line numbers increasing)
        # Examples: 1-2, 1-3, 1-2-3 (any sequence where last > previous and line 1 is present)
        # CRITICAL: This is checked AFTER exit detection, so if it was an exit, we already returned
        if last_crossing in [2, 3]:  # Entry ends with line 2 or line 3
            # Check if there's any previous crossing that's less than the last crossing
            # This means person moved from a lower line number to a higher one (inward movement)
            has_lower_previous = any(prev_line < last_crossing for prev_line in crossed_list[:-1])
            
            logger.info(f"Person {person.track_id} - Entry check: last_crossing={last_crossing}, has_lower_previous={has_lower_previous}, entered={person.entered}, exited={person.exited}")
            logger.info(f"  Full sequence: {crossed_list}")
            
            # Entry if: There's a previous crossing with lower line number (inward movement)
            if has_lower_previous:
                # NEW RULE: For any ENTRY, line 1 MUST be present somewhere in the history
                if 1 not in crossed_list:
                    logger.debug(f"Person {person.track_id} - Entry attempt ignored (line 1 not present in sequence). Sequence: {crossed_list}")
                    return
                # Prevent re-entry: If person already entered and hasn't exited, ignore
                if person.entered and not person.exited:
                    # Person already entered but hasn't exited yet - this is a re-entry attempt, ignore it
                    logger.debug(f"Person {person.track_id} - Re-entry attempt ignored (already entered, need to exit first). Sequence: {crossed_list}")
                    return
                
                # If person exited, they must cross line 1 again (go back inside) before new entry is counted
                if person.exited:
                    # Find the index where person exited (last time they crossed line 1)
                    exit_index = None
                    for i in range(len(crossed_list) - 1, -1, -1):
                        if crossed_list[i] == 1:
                            exit_index = i
                            break
                    
                    if exit_index is not None:
                        # Check if line 1 appears again after the exit (meaning they went back inside)
                        has_crossed_1_after_exit = 1 in crossed_list[exit_index + 1:-1]
                        if not has_crossed_1_after_exit:
                            # Person exited but hasn't crossed line 1 again - they're still outside
                            # This is a false entry (loitering near entrance), ignore it
                            logger.debug(f"Person {person.track_id} - False entry ignored (exited but hasn't crossed line 1 again). Sequence: {crossed_list}")
                            return
                        else:
                            # They crossed line 1 again after exiting - valid entry, reset entered flag
                            # Keep exited=True to track that they completed a cycle
                            person.entered = False  # Reset to allow new entry count
                            logger.debug(f"Person {person.track_id} - Entered after exit (crossed line 1 again), resetting entered flag for new entry")
                
                # Valid entry: Person has moved from lower line to higher line (inward movement)
                person.entered = True
                person.entry_time = time.time()
                self.entered_count += 1
                
                logger.info(f"Person {person.track_id} - ENTRY DETECTED! Sequence: {crossed_list} (inward movement), entered={person.entered}, exited={person.exited}")
                
                # Check for group entry
                self.recent_events['entered'].append((person.track_id, person.entry_time))
                self.recent_events['entered'] = [
                    (tid, ts) for tid, ts in self.recent_events['entered']
                    if person.entry_time - ts <= self.group_time_window
                ]
                
                simultaneous_entries = [
                    tid for tid, ts in self.recent_events['entered']
                    if tid != person.track_id and abs(person.entry_time - ts) <= self.group_time_window
                ]
                
                if len(simultaneous_entries) > 0:
                    person.is_in_group = True
                    self.group_entered += 1
                    for tid in simultaneous_entries:
                        if tid in self.tracked_people:
                            self.tracked_people[tid].is_in_group = True
                    logger.info(f"Person ID {person.track_id} ENTERED (GROUP) - Sequence: {crossed_list}")
                else:
                    person.is_in_group = False
                    self.individual_entered += 1
                    logger.info(f"Person ID {person.track_id} ENTERED (INDIVIDUAL) - Sequence: {crossed_list}")
                
                # Update gender/age stats
                if person.gender:
                    self.gender_stats[person.gender] = self.gender_stats.get(person.gender, 0) + 1
                else:
                    self.gender_stats['unknown'] += 1
                
                if person.age_group:
                    self.age_stats[person.age_group] = self.age_stats.get(person.age_group, 0) + 1
                else:
                    self.age_stats['unknown'] += 1
                return
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for detection, tracking, and counting.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Processed frame with visualizations
        """
        self.frame_count = getattr(self, 'frame_count', 0) + 1
        
        # Calculate filter line if not set and we have line_1
        if self.filter_line is None and self.lines and len(self.lines) >= 1:
            self.filter_line = self._calculate_filter_line(frame.shape)
            if self.filter_line:
                logger.info(f"Filter line calculated: {self.filter_line}")
        
        # Process every Nth frame for detection
        if self.frame_count % self.frame_skip == 0:
            raw_detections = self.yolo_client.detect_objects(frame, self.text_prompt)
            # Filter detections based on filter line (only keep detections on right side, where three lines are)
            if self.filter_line:
                filtered_detections = []
                (fl_x1, fl_y1), (fl_x2, fl_y2) = self.filter_line
                for x1, y1, x2, y2, conf, label in raw_detections:
                    # Get detection centroid
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Check which side of filter line the detection is on
                    side = self._point_to_line_side((cx, cy), self.filter_line)
                    
                    # Keep detections on RIGHT side (where three counting lines are)
                    # The three counting lines (line_1, line_2, line_3) are to the RIGHT of the filter line
                    if side == 'right':
                        filtered_detections.append((x1, y1, x2, y2, conf, label))
                raw_detections = filtered_detections
                logger.debug(f"Filtered detections: {len(filtered_detections)}/{len(self.last_raw_detections) if hasattr(self, 'last_raw_detections') else 0} on right side of filter line")
            self.last_raw_detections = raw_detections
        else:
            raw_detections = self.last_raw_detections
        
        # Convert to format expected by tracker
        detections_for_tracker = [(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf, _ in raw_detections]
        
        # Update tracker - ByteTracker uses different format
        if self.use_bytetracker and BYTETRACKER_AVAILABLE:
            # ByteTracker expects detections in format: numpy array [x1, y1, x2, y2, conf, class]
            # We need to add class_id (0 for person) to each detection
            if len(detections_for_tracker) > 0:
                det_array = np.array([list(d) + [0.0] for d in detections_for_tracker], dtype=np.float32)
                # ByteTracker.update(frame, detections) returns tracks
                # Format: [x1, y1, x2, y2, track_id, conf, class]
                tracks = self.tracker.update(det_array, frame)
                # Convert to our format: (x1, y1, x2, y2, conf, track_id)
                tracked_detections = []
                for t in tracks:
                    if len(t) >= 7:
                        # ByteTracker returns: [x1, y1, x2, y2, track_id, conf, class]
                        x1, y1, x2, y2, track_id, conf, cls = t[0], t[1], t[2], t[3], t[4], t[5], t[6]
                        tracked_detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(track_id)))
            else:
                # No detections - ByteTracker still needs to be updated
                empty_det = np.empty((0, 6), dtype=np.float32)
                tracks = self.tracker.update(empty_det, frame)
                tracked_detections = []
        else:
            # ImprovedTracker format
            tracked_detections = self.tracker.update(detections_for_tracker)
        
        # Map labels from raw detections to tracked objects
        self.track_id_to_label = {}
        for x1, y1, x2, y2, conf, track_id in tracked_detections:
            track_bbox = (x1, y1, x2, y2)
            best_iou = 0
            best_label = "people"
            
            for rx1, ry1, rx2, ry2, rconf, rlabel in raw_detections:
                raw_bbox = (rx1, ry1, rx2, ry2)
                iou = self._calculate_iou(track_bbox, raw_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_label = rlabel
            
            if best_iou > 0.3:
                self.track_id_to_label[track_id] = best_label
        
        # Draw filter line if it exists
        if self.filter_line:
            (x1, y1), (x2, y2) = self.filter_line
            # Draw filter line in magenta/dashed style to distinguish from counting lines
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2, cv2.LINE_AA)
            # Draw label
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(
                frame,
                "FILTER LINE (LEFT SIDE IGNORED)",
                (mid_x - 150, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2
            )
        
        # Draw three straight lines or circular (radial) zones
        if self.lines and len(self.lines) == 3:
            if self.use_circular_zones and self.circle_center and self.circle_radii:
                # Draw three concentric half-circles (upper half) for entry/exit zones
                cx, cy = int(self.circle_center[0]), int(self.circle_center[1])
                line_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0)]  # Green, Yellow, Red
                line_labels = ["ZONE 1", "ZONE 2", "ZONE 3"]
                
                for radius, color, label in zip(self.circle_radii, line_colors, line_labels):
                    r = int(radius)
                    # Draw a half-circle (180° arc). Adjust angles if needed (here: 0 to 180 degrees).
                    cv2.ellipse(
                        frame,
                        (cx, cy),
                        (r, r),
                        0,          # rotation
                        0, 180,     # startAngle, endAngle (upper half-circle)
                        color,
                        3,
                        cv2.LINE_AA
                    )
                    # Label slightly above the arc
                    cv2.putText(
                        frame,
                        label,
                        (cx - r, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
            else:
                # Default: draw the three straight counting lines
                line_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0)]  # Green, Yellow, Red
                line_labels = ["LINE 1 (LEFT)", "LINE 2 (MIDDLE)", "LINE 3 (RIGHT)"]
                
                for i, (line, color, label) in enumerate(zip(self.lines, line_colors, line_labels)):
                    (x1, y1), (x2, y2) = line
                    cv2.line(frame, (x1, y1), (x2, y2), color, 3)
                    # Draw label at line midpoint
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    cv2.putText(
                        frame,
                        label,
                        (mid_x - 80, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
        
        # Draw raw detections
        if self.last_raw_detections:
            for x1, y1, x2, y2, conf, label in self.last_raw_detections:
                if conf >= self.confidence:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 100), 1)
        
        # Process tracked detections
        current_people: Dict[int, PersonInfo] = {}
        
        for x1, y1, x2, y2, conf, track_id in tracked_detections:
            # Allow predicted tracks (low confidence) to continue processing for line crossing
            # Only skip if confidence is too low (predicted tracks have conf=0.3)
            is_predicted = conf < self.confidence and conf >= 0.2
            if conf < 0.2:  # Skip only very low confidence detections
                continue
            
            # Validate bbox
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bbox for track {track_id}: ({x1}, {y1}, {x2}, {y2})")
                continue
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centroid = (cx, cy)
            
            # Validate centroid
            if not (isinstance(cx, (int, float)) and isinstance(cy, (int, float))):
                logger.warning(f"Invalid centroid for track {track_id}: ({cx}, {cy})")
                continue
            
            # Get or create person info
            if track_id not in self.tracked_people:
                label = self.track_id_to_label.get(track_id, "people")
                person = PersonInfo(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    centroid=centroid,
                    trajectory=deque(maxlen=30),  # Keep last 30 positions
                    entered=False,
                    exited=False,
                    label=label,
                    lines_crossed=deque(maxlen=20),  # Keep last 20 line crossings
                    line_sides={}
                )
                self.tracked_people[track_id] = person
                logger.info(f"New person {track_id} created at position {centroid}")
            else:
                person = self.tracked_people[track_id]
                # Initialize line_sides if not exists
                if person.line_sides is None:
                    person.line_sides = {}
                
                # CRITICAL: Verify trajectory is not being reset
                # The trajectory should accumulate across all frames
                if len(person.trajectory) == 0:
                    logger.warning(f"Person {track_id} - Trajectory was empty! Restoring from last centroid.")
                    # Restore trajectory if it was somehow lost (shouldn't happen, but safety check)
                    if person.centroid:
                        person.trajectory.append(person.centroid)
                        logger.info(f"Person {track_id} - Restored trajectory with centroid {person.centroid}")
            
            # Update label if we have a new one
            if track_id in self.track_id_to_label:
                person.label = self.track_id_to_label[track_id]
            
            # Update person info - IMPORTANT: Update trajectory BEFORE checking line crossings
            # This ensures we have the latest position for line crossing detection
            prev_centroid = person.centroid if len(person.trajectory) > 0 else None
            prev_trajectory_length = len(person.trajectory)
            
            # Update bbox and centroid
            person.bbox = (x1, y1, x2, y2)
            person.centroid = centroid
            
            # CRITICAL: Append to trajectory - this MUST happen for every detection
            # The trajectory must accumulate across all frames (including predicted ones)
            # Note: deque with maxlen will automatically drop old items when full, so length may stay at maxlen
            try:
                person.trajectory.append(centroid)
                person.last_seen = 0
                
                # Verify trajectory was actually added
                # For deque with maxlen=30, when full, length stays at 30 (old items are dropped)
                # So we only check if length increased OR if it's at maxlen (which is also valid)
                max_trajectory_length = 30  # Should match deque maxlen
                if len(person.trajectory) < prev_trajectory_length:
                    # This should never happen - trajectory length decreased
                    logger.error(f"Person {person.track_id} - Trajectory length decreased! Was {prev_trajectory_length}, now {len(person.trajectory)}")
                    # Force add if it somehow decreased
                    person.trajectory.append(centroid)
                    logger.info(f"Person {person.track_id} - Force added trajectory point after decrease")
                elif len(person.trajectory) == prev_trajectory_length and prev_trajectory_length < max_trajectory_length:
                    # Length didn't increase when it should have (and we're not at max)
                    logger.warning(f"Person {person.track_id} - Trajectory append may have failed. Length stayed at {prev_trajectory_length}")
                    # Try to add again
                    person.trajectory.append(centroid)
                # If length == max_trajectory_length, that's normal - deque is full and dropping old items
            except Exception as e:
                logger.error(f"Person {person.track_id} - Error appending to trajectory: {e}")
                # Try to recover by creating new deque if needed
                if not hasattr(person, 'trajectory') or person.trajectory is None:
                    person.trajectory = deque(maxlen=30)
                person.trajectory.append(centroid)
            
            # Debug: Log trajectory updates for tracking issues
            if prev_trajectory_length == 0 and len(person.trajectory) == 1:
                logger.info(f"Person {person.track_id} - First trajectory point added: {centroid}")
            elif len(person.trajectory) >= 2:
                # Check if trajectory is actually moving
                traj_list = list(person.trajectory)
                prev_point = traj_list[-2]
                curr_point = traj_list[-1]
                distance = np.sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)
                if distance < 1.0 and not is_predicted:
                    logger.debug(f"Person {person.track_id} - Trajectory not moving much: {distance:.2f} pixels")
            
            # Detect gender and age (every 5 frames to reduce computation)
            if self.gender_age_detector and (person.gender is None or person.age_group is None):
                # Run detection every 5 frames or when person first appears
                if len(person.trajectory) % 5 == 0 or len(person.trajectory) == 1:
                    gender, age_group = self.gender_age_detector.detect_gender_age(frame, (x1, y1, x2, y2))
                    if gender:
                        person.gender = gender
                        logger.debug(f"Person {person.track_id} - Gender detected: {gender}")
                    if age_group:
                        person.age_group = age_group
                        logger.debug(f"Person {person.track_id} - Age group detected: {age_group}")
            
            # Check line crossings - track which lines/zones have been crossed in order
            if self.lines and len(self.lines) == 3:
                # Initialize lines_crossed if needed
                if person.lines_crossed is None:
                    person.lines_crossed = deque(maxlen=20)  # Increased to keep more history
                
                # Track which side of each line the person is currently on
                if not hasattr(person, 'line_sides'):
                    person.line_sides = {}  # {line_num: 'left' or 'right'}
                
                # Track last checked trajectory index to avoid re-checking same segments
                if not hasattr(person, 'last_checked_trajectory_index'):
                    person.last_checked_trajectory_index = 0
                
                # Check each line or circular boundary for crossing - after updating trajectory
                if len(person.trajectory) >= 2:
                    traj_list = list(person.trajectory)
                    start_idx = max(0, person.last_checked_trajectory_index - 1)  # Check from slightly before last check
                    
                    if self.use_circular_zones and self.circle_center and self.circle_radii:
                        # Circular mode: use radial distance from center to detect boundary crossings
                        cx0, cy0 = self.circle_center
                        
                        def zone_index(pt: Tuple[float, float]) -> int:
                            """Return zone index 0..3 for a point based on distance to center."""
                            dx = pt[0] - cx0
                            dy = pt[1] - cy0
                            r = np.sqrt(dx * dx + dy * dy)
                            r1, r2, r3 = self.circle_radii
                            if r < r1:
                                return 0
                            elif r < r2:
                                return 1
                            elif r < r3:
                                return 2
                            else:
                                return 3
                        
                        # Check trajectory segments for zone boundary crossings
                        for i in range(start_idx, len(traj_list) - 1):
                            prev_point = traj_list[i]
                            current_point = traj_list[i + 1]
                            
                            if prev_point is None or current_point is None:
                                continue
                            
                            z_prev = zone_index(prev_point)
                            z_curr = zone_index(current_point)
                            
                            if z_prev == z_curr:
                                continue
                            
                            # Determine which boundaries (1..3) were crossed between zones
                            if z_prev < z_curr:
                                # Moving outward: cross boundaries z_prev+1 .. z_curr
                                boundaries = range(z_prev + 1, min(z_curr, 3) + 1)
                            else:
                                # Moving inward: cross boundaries z_curr+1 .. z_prev (reverse order)
                                boundaries = range(z_prev, max(z_curr, 0), -1)
                            
                            for line_num in boundaries:
                                if line_num < 1 or line_num > 3:
                                    continue
                                recent_crossings = list(person.lines_crossed)[-3:] if len(person.lines_crossed) >= 3 else list(person.lines_crossed)
                                if line_num not in recent_crossings:
                                    logger.info(f"Person {person.track_id} crossed circular boundary {line_num} at frame {self.frame_count}")
                                    logger.info(f"  Movement: {prev_point} -> {current_point}, zones: {z_prev}->{z_curr}")
                                    logger.info(f"  Previous crossings: {list(person.lines_crossed)}")
                                    person.lines_crossed.append(line_num)
                                    logger.info(f"  Updated crossings: {list(person.lines_crossed)}")
                                    self._check_entry_exit(person, line_num)
                            # Note: do not break; a segment can cross multiple boundaries
                    else:
                        # Default line mode: use straight line crossing detection
                        # Check all trajectory segments since last check (not just last 2 points)
                        for line in self.lines:
                            line_num = self._get_line_number(line)
                            if line_num == 0:
                                continue
                            
                            # Check trajectory segments for line crossing
                            # Check from start_idx to end, checking pairs of points
                            for i in range(start_idx, len(traj_list) - 1):
                                prev_point = traj_list[i]
                                current_point = traj_list[i + 1]
                                
                                # Verify points are valid
                                if prev_point is None or current_point is None:
                                    continue
                                
                                # Check if movement from prev to current crosses this line segment
                                if self._has_crossed_line(prev_point, current_point, line):
                                    # Check if this line was already recorded in recent crossings
                                    # Use last 3 crossings to prevent duplicates
                                    recent_crossings = list(person.lines_crossed)[-3:] if len(person.lines_crossed) >= 3 else list(person.lines_crossed)
                                    
                                    # Only record if this line number is not in the last few crossings
                                    # This prevents duplicate detections when person is stationary near a line
                                    if line_num not in recent_crossings:
                                        logger.info(f"Person {person.track_id} crossed line {line_num} at frame {self.frame_count}")
                                        logger.info(f"  Movement: {prev_point} -> {current_point}")
                                        logger.info(f"  Previous crossings: {list(person.lines_crossed)}")
                                        person.lines_crossed.append(line_num)
                                        logger.info(f"  Updated crossings: {list(person.lines_crossed)}")
                                        # Check for entry/exit sequence
                                        self._check_entry_exit(person, line_num)
                                        break  # Found crossing for this line, move to next line
                    
                    # Update last checked index
                    person.last_checked_trajectory_index = len(traj_list) - 1
                
                # Also track current side for visualization/debugging
                for line in self.lines:
                    line_num = self._get_line_number(line)
                    if line_num != 0:
                        current_side = self._point_to_line_side(centroid, line)
                        person.line_sides[line_num] = current_side
            
            current_people[track_id] = person
            
            # Determine color based on status
            if person.entered and person.exited:
                color = (0, 255, 255)  # Yellow
                status = "ENTERED & EXITED"
            elif person.entered:
                color = (0, 255, 0)  # Green
                status = "ENTERED"
            elif person.exited:
                color = (255, 0, 255)  # Magenta
                status = "EXITED"
            else:
                color = (255, 0, 0)  # Red
                status = "TRACKING"
            
            # Draw bounding box (thinner/dashed style for predicted tracks)
            if is_predicted:
                # Draw thinner rectangle for predicted tracks to indicate they're predicted
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Add "PREDICTED" indicator
                status = status + " [PREDICTED]"
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with status and info
            group_indicator = " [GROUP]" if person.is_in_group else ""
            gender_text = f" [{person.gender.upper()}]" if person.gender else ""
            age_text = f" [{person.age_group.upper()}]" if person.age_group else ""
            
            # Show lines crossed
            lines_crossed_text = ""
            if person.lines_crossed and len(person.lines_crossed) > 0:
                recent_lines = list(person.lines_crossed)[-5:]  # Last 5 crossings
                lines_crossed_text = f" Lines: {recent_lines}"
            
            label_text = f"{person.label.upper()} ID:{track_id} {status}{group_indicator}{gender_text}{age_text}{lines_crossed_text}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 4, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label_text,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),  # Black text
                2
            )
            
            # Draw trajectory
            if len(person.trajectory) > 1:
                points = list(person.trajectory)
                for i in range(1, len(points)):
                    cv2.line(frame, 
                            (int(points[i-1][0]), int(points[i-1][1])),
                            (int(points[i][0]), int(points[i][1])),
                            color, 2)
            
            # Draw direction arrow
            if len(person.trajectory) >= 2:
                prev_pos = person.trajectory[-2]
                curr_pos = person.trajectory[-1]
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                if abs(dx) > 1 or abs(dy) > 1:
                    cv2.arrowedLine(frame,
                                   (int(prev_pos[0]), int(prev_pos[1])),
                                   (int(curr_pos[0]), int(curr_pos[1])),
                                   color, 2, tipLength=0.3)
        
        # Remove old tracks
        tracks_to_remove = [tid for tid in self.tracked_people.keys() if tid not in current_people]
        for tid in tracks_to_remove:
            person = self.tracked_people[tid]
            person.last_seen += 1
            if person.last_seen > 5:
                del self.tracked_people[tid]
        
        # Draw statistics overlay
        self._draw_statistics(frame)
        
        return frame
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _draw_statistics(self, frame: np.ndarray):
        """Draw statistics overlay on frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 400), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw statistics
        stats_text = [
            f"TOTAL ENTERED: {self.entered_count}",
            f"  - Individual: {self.individual_entered} | Group: {self.group_entered}",
            f"TOTAL EXITED: {self.exited_count}",
            f"  - Individual: {self.individual_exited} | Group: {self.group_exited}",
            "",
            "GENDER STATS:",
            f"  - Male: {self.gender_stats.get('male', 0)} | Female: {self.gender_stats.get('female', 0)} | Unknown: {self.gender_stats.get('unknown', 0)}",
            "",
            "AGE STATS:",
            f"  - Child: {self.age_stats.get('child', 0)} | Young: {self.age_stats.get('young_adult', 0)}",
            f"  - Middle: {self.age_stats.get('middle_aged', 0)} | Senior: {self.age_stats.get('senior', 0)} | Unknown: {self.age_stats.get('unknown', 0)}",
        ]
        
        y_offset = 40
        line_height = 20
        
        for i, text in enumerate(stats_text):
            if not text:
                continue
            if i < 2:
                color = (0, 255, 0)  # Green for entered
            elif i < 4:
                color = (255, 0, 0)  # Red for exited
            else:
                color = (200, 200, 200)  # Gray for stats
            
            cv2.putText(
                frame,
                text,
                (20, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6 if i < 4 else 0.5,
                color,
                2 if i < 4 else 1
            )
        
        # Add model info
        api_stats = self.yolo_client.get_stats()
        model_info = [
            f"YOLOv8 Detections: {api_stats['total_requests']}",
            f"Success Rate: {api_stats['success_rate']:.1f}%",
            f"Frame Skip: {self.frame_skip}",
        ]
        
        y_offset = 350
        for i, text in enumerate(model_info):
            cv2.putText(
                frame,
                text,
                (20, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        api_stats = self.yolo_client.get_stats()
        return {
            'entered': self.entered_count,
            'exited': self.exited_count,
            'individual_entered': self.individual_entered,
            'group_entered': self.group_entered,
            'individual_exited': self.individual_exited,
            'group_exited': self.group_exited,
            'gender_stats': self.gender_stats.copy(),
            'age_stats': self.age_stats.copy(),
            'api_stats': api_stats
        }

