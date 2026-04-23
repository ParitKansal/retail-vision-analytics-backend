"""
Service-specific utilities for Entry/Exit Counting
Contains ThreeLineCounter and related classes
"""

import cv2
import numpy as np
import logging
import time
import math
import urllib.request
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

# Try to import MTCNN
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.warning("MTCNN not available. Install with: pip install mtcnn")

# Default paths
ARTIFACTS_DIR = Path(__file__).parent.resolve() / "artifacts"
DEFAULT_MODEL_PATH = str(ARTIFACTS_DIR / "yolov8l.pt")


class YOLOWorldClient:
    """Client for local YOLOv8 model with Native ByteTrack Support."""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, confidence: float = 0.3):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available.")

        model_path = Path(model_path)
        if not model_path.is_absolute():
            current_dir = Path(__file__).parent
            model_path = current_dir / model_path

        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}, letting Ultralytics handle download...")
            model_path = str(model_path) 

        logger.info(f"Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.request_count = 0

    def track_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, int, float]]:
        """Run YOLOv8 Native Tracking (ByteTrack)."""
        try:
            self.request_count += 1
            # persist=True is critical for ID tracking
            results = self.model.track(
                source=frame, 
                persist=True, 
                conf=self.confidence, 
                tracker="bytetrack.yaml",
                classes=[0], # 0 is Person class
                verbose=False
            )

            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()

                    for box, track_id, conf in zip(boxes, track_ids, confs):
                        x1, y1, x2, y2 = map(int, box[:4])
                        detections.append((x1, y1, x2, y2, int(track_id), float(conf)))

            return detections
        except Exception as e:
            logger.error(f"Error in track_objects: {e}")
            return []
    
    def get_stats(self) -> Dict:
        return {"total_requests": self.request_count}


class GenderAgeDetector:
    """Lightweight Gender and Age detection using OpenCV DNN (Caffe) + MTCNN."""
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # Mapping output index to readable labels
    AGE_LIST = ['child', 'child', 'child', 'young_adult', 'young_adult', 'middle_aged', 'middle_aged', 'senior']
    GENDER_LIST = ['male', 'female']
    
    FILES = {
        "age_deploy.prototxt": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt",
        "age_net.caffemodel": "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel",
        "gender_deploy.prototxt": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt",
        "gender_net.caffemodel": "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel"
    }

    def __init__(self):
        self.model_available = False
        self.artifacts_dir = ARTIFACTS_DIR
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        if MTCNN_AVAILABLE:
            try:
                # Optimized MTCNN params for speed
                self.face_detector = MTCNN(min_face_size=20, steps_threshold=[0.7, 0.8, 0.8])
            except Exception: return
        else: return

        if not self._ensure_models(): return

        try:
            self.age_net = cv2.dnn.readNet(str(self.artifacts_dir / "age_deploy.prototxt"), str(self.artifacts_dir / "age_net.caffemodel"))
            self.gender_net = cv2.dnn.readNet(str(self.artifacts_dir / "gender_deploy.prototxt"), str(self.artifacts_dir / "gender_net.caffemodel"))
            
            # Use CPU optimization (OpenCV DNN is very fast on CPU)
            for net in [self.age_net, self.gender_net]:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.model_available = True
            logger.info("Lightweight Caffe Gender/Age models loaded.")
        except Exception as e:
            logger.error(f"Failed to load Caffe models: {e}")

    def _ensure_models(self) -> bool:
        try:
            for filename, url in self.FILES.items():
                file_path = self.artifacts_dir / filename
                if not file_path.exists():
                    logger.info(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, str(file_path))
            return True
        except Exception: return False

    def detect_gender_age(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], Optional[str]]:
        if not self.model_available: return None, None
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        # Add context padding
        pad_x, pad_y = int((x2-x1)*0.1), int((y2-y1)*0.1)
        roi = frame[max(0, y1-pad_y):min(h, y2+pad_y), max(0, x1-pad_x):min(w, x2+pad_x)]
        
        if roi.size == 0 or roi.shape[0] < 30: return None, None
        
        try:
            # Detect face in ROI
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            faces = self.face_detector.detect_faces(rgb_roi)
            if not faces: return None, None
            
            # Get largest face
            fx, fy, fw, fh = max(faces, key=lambda f: f['box'][2]*f['box'][3])['box']
            face_img = roi[max(0, fy-10):fy+fh+10, max(0, fx-10):fx+fw+10]
            
            if face_img.size == 0: return None, None
            
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            
            self.gender_net.setInput(blob)
            gender = self.GENDER_LIST[self.gender_net.forward()[0].argmax()]
            
            self.age_net.setInput(blob)
            age = self.AGE_LIST[self.age_net.forward()[0].argmax()]
            
            return gender, age
        except Exception: return None, None


@dataclass
class PersonInfo:
    track_id: int
    bbox: Tuple[int, int, int, int]
    tracking_point: Tuple[float, float]
    trajectory: deque
    last_seen: int = 0
    entered: bool = False
    exited: bool = False
    entry_time: float = 0.0
    exit_time: float = 0.0
    is_in_group: bool = False
    gender: Optional[str] = None
    age_group: Optional[str] = None
    lines_crossed: deque = None


class ThreeLineCounter:
    """Uses YOLOv8 Native Tracking (ByteTrack) + Three Line Logic + Explicit Filter Line."""
    
    def __init__(self, lines: List, model_path: str, confidence: float, enable_gender_age: bool, filter_line: Optional[Tuple] = None, **kwargs):
        self.lines = lines  # [Line 1, Line 2, Line 3]
        self.confidence = confidence
        self.filter_line = filter_line 
        self.valid_side = None # 'left' or 'right'
        
        self.yolo_client = YOLOWorldClient(model_path=model_path, confidence=confidence)
        self.gender_age_detector = GenderAgeDetector() if enable_gender_age else None
        
        self.tracked_people: Dict[int, PersonInfo] = {}
        self.frame_count = 0
        
        self.entered_count = 0
        self.exited_count = 0
        self.gender_stats = {'male': 0, 'female': 0, 'unknown': 0}
        self.age_stats = {k: 0 for k in GenderAgeDetector.AGE_LIST}
        self.age_stats['unknown'] = 0
        
        # Initialize Valid Side Logic if filter line exists
        if self.filter_line and len(self.lines) > 0:
            self._determine_valid_side()

    def _get_line_number(self, line): return self.lines.index(line) + 1 if line in self.lines else 0

    def _point_to_line_side(self, p, l): 
        # Standard Cross Product
        val = (l[1][0]-l[0][0])*(p[1]-l[0][1]) - (l[1][1]-l[0][1])*(p[0]-l[0][0])
        return 'right' if val < 0 else 'left'

    def _determine_valid_side(self):
        """Determines valid tracking side relative to Filter Line (using Line 1 position)"""
        if not self.filter_line or not self.lines: return
        
        l1_start, l1_end = self.lines[0]
        mid_l1 = ((l1_start[0]+l1_end[0])/2, (l1_start[1]+l1_end[1])/2)
        
        self.valid_side = self._point_to_line_side(mid_l1, self.filter_line)
        logger.info(f"Valid Tracking Side: {self.valid_side}")

    def _check_entry_exit(self, person: PersonInfo, frame: np.ndarray):
        if len(person.lines_crossed) < 2: return
        seq = list(person.lines_crossed)
        last = seq[-1]
        
        # === EXIT LOGIC (3 -> 2 -> 1) ===
        if last == 1 and any(x > 1 for x in seq[:-1]):
            if not person.exited and not person.entered:
                person.exited = True
                person.exit_time = time.time()
                self.exited_count += 1
                return

        # === ENTRY LOGIC (1 -> 2 -> 3) OR (2 -> 3) ===
        if last == 3 and (2 in seq or 1 in seq):
            if not person.entered and not person.exited:
                person.entered = True
                person.entry_time = time.time()
                self.entered_count += 1
                
                # --- FORCE DEMOGRAPHICS ON ENTRY ---
                # This ensures we get Age/Gender NOW to visualize along with Entry
                if self.gender_age_detector:
                    g, a = self.gender_age_detector.detect_gender_age(frame, person.bbox)
                    if g: person.gender = g
                    if a: person.age_group = a
                    if not g: logger.warning(f"ID {person.track_id}: Face not detected at entry.")
                
                # Update Stats
                gender_key = person.gender if person.gender else 'unknown'
                age_key = person.age_group if person.age_group else 'unknown'
                
                self.gender_stats[gender_key] = self.gender_stats.get(gender_key, 0) + 1
                self.age_stats[age_key] = self.age_stats.get(age_key, 0) + 1
                return

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        
        tracked_objects = self.yolo_client.track_objects(frame)
        current_ids = set()
        
        for x1, y1, x2, y2, track_id, conf in tracked_objects:
            
            # Tracking Point: Bottom Center (Feet)
            center_x = (x1 + x2) / 2
            bottom_y = float(y2)
            tracking_point = (center_x, bottom_y)
            
            # --- FILTER CHECK ---
            if self.filter_line and self.valid_side:
                side = self._point_to_line_side(tracking_point, self.filter_line)
                if side != self.valid_side:
                    continue

            current_ids.add(track_id)
            
            if track_id not in self.tracked_people:
                person = PersonInfo(track_id, (x1,y1,x2,y2), tracking_point, deque(maxlen=30), lines_crossed=deque(maxlen=20))
                self.tracked_people[track_id] = person
            else:
                person = self.tracked_people[track_id]
                person.bbox = (x1,y1,x2,y2)
                person.tracking_point = tracking_point
                person.trajectory.append(tracking_point)
                person.last_seen = 0
            
            # Periodic Background Check (e.g., if force detection failed or track is long)
            if self.gender_age_detector and (person.gender is None or self.frame_count % 15 == 0):
                if len(person.trajectory) > 5:
                    g, a = self.gender_age_detector.detect_gender_age(frame, (x1, y1, x2, y2))
                    if g: person.gender = g
                    if a: person.age_group = a

            # Line Crossing
            if len(person.trajectory) >= 2:
                for line in self.lines:
                    lnum = self._get_line_number(line)
                    prev, curr = person.trajectory[-2], person.trajectory[-1]
                    
                    if self._point_to_line_side(prev, line) != self._point_to_line_side(curr, line):
                        if not person.lines_crossed or person.lines_crossed[-1] != lnum:
                            person.lines_crossed.append(lnum)
                            # Pass frame for immediate demographics
                            self._check_entry_exit(person, frame)

            # --- VISUALIZATION ---
            # Green = Entered, Red = Exited, Blue = Tracking
            color = (0, 255, 0) if person.entered else ((0, 0, 255) if person.exited else (255, 0, 0))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(tracking_point[0]), int(tracking_point[1])), 5, (0, 255, 255), -1)
            
            # Format Label: ID GENDER AGE [EVENT]
            status = ""
            if person.entered: status = "[ENTRY]"
            elif person.exited: status = "[EXIT]"
            
            g_label = (person.gender or "?").upper()
            a_label = (person.age_group or "?").upper()
            
            label = f"ID:{track_id} {g_label} {a_label} {status}"
            
            # Draw Label with background for readability
            (w_text, h_text), _ = cv2.getTextSize(label, 0, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-20), (x1+w_text, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), 0, 0.5, (255, 255, 255), 1)

        # Draw Lines
        for i, line in enumerate(self.lines):
            cv2.line(frame, line[0], line[1], (0, 255, 255), 2)
            cv2.putText(frame, f"L{i+1}", line[0], 0, 0.5, (0,255,255), 2)
        
        # Draw Filter Line
        if self.filter_line:
            pt1 = (int(self.filter_line[0][0]), int(self.filter_line[0][1]))
            pt2 = (int(self.filter_line[1][0]), int(self.filter_line[1][1]))
            cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
            mid_f = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(frame, "FILTER", (mid_f[0]-40, mid_f[1]), 0, 0.6, (255,0,255), 2)

        for tid in list(self.tracked_people.keys()):
            if tid not in current_ids:
                self.tracked_people[tid].last_seen += 1
                if self.tracked_people[tid].last_seen > 30:
                    del self.tracked_people[tid]

        return frame

    def get_statistics(self) -> Dict:
        return {
            'entered': self.entered_count,
            'exited': self.exited_count,
            'gender_stats': self.gender_stats,
            'age_stats': self.age_stats
        }