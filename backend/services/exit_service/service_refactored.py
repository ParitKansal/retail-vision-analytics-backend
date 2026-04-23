import json
import numpy as np
import cv2
from dotenv import load_dotenv
from pathlib import Path
import redis
import logging
import time
import requests
import logging_loki
from multiprocessing import Queue
import os
import base64
from collections import deque, Counter, defaultdict
import uuid
import urllib.request
import yaml

# AI Models
try:
    from ultralytics.trackers import BYTETracker
except ImportError:
    print("WARNING: ultralytics ByteTracker not found. Please install ultralytics.")
    BYTETracker = None

from mtcnn import MTCNN

load_dotenv()

# --- CONFIGURATION ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

# Stream Config
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
TARGET_FPS = int(os.getenv("TARGET_FPS", "5"))

# Model paths - use local files from service directory
SERVICE_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = Path("/workspace")

# Check workspace first (Docker), then service directory (local dev)
def get_model_path(filename, env_var=None, default=None):
    """Find model file in workspace or service directory."""
    workspace_path = WORKSPACE_DIR / filename
    service_path = SERVICE_DIR / filename
    
    if workspace_path.exists():
        return str(workspace_path)
    elif service_path.exists():
        return str(service_path)
    elif env_var:
        return os.getenv(env_var, default)
    return None

ONNX_MODEL_PATH = get_model_path("emotion-ferplus-8.onnx")


# Output
OUTPUT_QUEUE_1 = os.getenv("OUTPUT_QUEUE_NAME_1", "emotion_alerts")
TARGET_COLLECTION = os.getenv("TARGET_COLLECTION", "emotion_events")
CAMERA_ID = os.getenv("CAMERA_ID", "smilometer")
VISUALIZE = str(os.getenv("VISUALIZE", "False")).lower() == "true"
YOLO_MODEL_URL = os.getenv("YOLO_MODEL_URL", "http://yolo_service:8082/predict")

# Constants
CONFIDENCE_THRESHOLD = 0.6
EMOTIONS_LABELS = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']

# VOTING CONFIG
VOTE_COUNT = 3  # Number of frames to collect before deciding the emotion

# --- LOGGING SETUP ---
try:
    handler = logging_loki.LokiQueueHandler(
        Queue(-1),
        url=LOKI_URL,
        tags={"application": "direct_emotion_service"},
        version="1",
    )
    logger = logging.getLogger("direct_emotion_service")
    logger.addHandler(handler)
except Exception:
    logger = logging.getLogger("direct_emotion_service")

logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class MockBoxes:
    """
    Mock class to mimic ultralytics Boxes object.
    BYTETracker expects results.boxes with .xyxy, .conf, .cls, .xywh attributes.
    """

    def __init__(self, detections: np.ndarray):
        """
        Args:
            detections: numpy array of shape (N, 6) with [x1, y1, x2, y2, conf, cls]
        """
        self._detections = detections
        if len(detections) == 0:
            self.xyxy = np.empty((0, 4))
            self.conf = np.empty((0,))
            self.cls = np.empty((0,))
            self.xywh = np.empty((0, 4))
        else:
            self.xyxy = detections[:, :4]  # [x1, y1, x2, y2]
            self.conf = detections[:, 4]  # confidence scores
            self.cls = detections[:, 5]  # class IDs
            # Convert xyxy to xywh (center_x, center_y, width, height)
            self.xywh = self._xyxy_to_xywh(self.xyxy)

    @staticmethod
    def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [center_x, center_y, width, height]."""
        if len(xyxy) == 0:
            return np.empty((0, 4))
        xywh = np.zeros_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  # center_x
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  # center_y
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # width
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # height
        return xywh

    def __len__(self):
        return len(self._detections)

    def __getitem__(self, idx):
        """Support indexing/slicing for BYTETracker filtering operations."""
        if isinstance(idx, (int, slice)):
            sliced_dets = self._detections[idx]
        else:
            sliced_dets = self._detections[idx]
        if sliced_dets.ndim == 1:
            sliced_dets = sliced_dets.reshape(1, -1)
        return MockBoxes(sliced_dets)


class MockResults:
    """
    Mock class to mimic ultralytics Results object for BYTETracker compatibility.
    """

    def __init__(self, detections: np.ndarray, orig_shape: tuple):
        self._detections = detections
        self.boxes = MockBoxes(detections)
        self.orig_shape = orig_shape

        if len(detections) == 0:
            self.conf = np.empty((0,))
            self.cls = np.empty((0,))
            self.xyxy = np.empty((0, 4))
            self.xywh = np.empty((0, 4))
        else:
            self.conf = detections[:, 4]
            self.cls = detections[:, 5]
            self.xyxy = detections[:, :4]
            self.xywh = self.boxes.xywh

    def __len__(self):
        return len(self._detections)

    def __getitem__(self, idx):
        """Support indexing/slicing for BYTETracker filtering operations."""
        if isinstance(idx, (int, slice)):
            sliced_dets = self._detections[idx]
        else:
            sliced_dets = self._detections[idx]
        if sliced_dets.ndim == 1:
            sliced_dets = sliced_dets.reshape(1, -1)
        return MockResults(sliced_dets, self.orig_shape)


class FixedSizeQueue:
    def __init__(self, redis_client, name, max_size):
        self.r = redis_client
        self.name = name
        self.max_size = max_size

    def push(self, item):
        """
        Push item to queue.
        Oldest items are dropped.
        """
        pipe = self.r.pipeline()
        pipe.rpush(self.name, item)
        pipe.ltrim(self.name, -self.max_size, -1)
        pipe.execute()

    def pop(self, timeout=0):
        """
        Blocking FIFO pop.
        """
        result = self.r.blpop(self.name, timeout=timeout)
        if result:
            _, value = result
            return value
        return None

    def size(self):
        return self.r.llen(self.name)


class DirectEmotionService:
    def __init__(self):
        self.settings = self.load_settings()
        self.redis = self.connect_redis()
        
        # 1. COMPLETED: IDs that have already been pushed to Redis
        self.processed_track_ids = set()
        
        # 2. BUFFER: Temporary storage for voting { track_id: [emotion1, emotion2, ...] }
        self.track_emotion_buffer = {}

        self.service_url = YOLO_MODEL_URL
        self.tracker = None
        self.init_tracker()

        self.mtcnn_detector = MTCNN()
        self.emotion_net = self.load_onnx_model()
        
        # Setup visualization queue if enabled
        if VISUALIZE:
            self.vis_queue = FixedSizeQueue(
                self.redis, f"stream:visualization:{CAMERA_ID}", 10
            )
            logger.info(f"Visualization queue initialized: stream:visualization:{CAMERA_ID}")
        
    def load_settings(self):
        base_dir = Path(__file__).parent.resolve()
        settings_path = base_dir / "setting.json"
        try:
            with open(settings_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return {}

    def connect_redis(self):
        while True:
            try:
                r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
                r.ping()
                logger.info("Connected to Redis output.")
                return r
            except Exception as e:
                logger.error(f"Redis failed: {e}. Retrying in 5s...")
                time.sleep(5)

    def init_tracker(self):
        """Initialize local ByteTracker with defaults."""
        if BYTETracker:
            # Default values 
            tracker_config = {
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
                "frame_rate": 30,
            }
            # Create arguments object for BYTETracker
            class TrackerArgs:
                def __init__(self, config):
                    for k, v in config.items():
                        setattr(self, k, v)
                    # Add missing required args that might be accessed
                    self.mot20 = False

            self.tracker = BYTETracker(TrackerArgs(tracker_config))
            logger.info("Local ByteTracker initialized.")
        else:
            logger.error("ByteTracker could not be initialized!")

    def load_onnx_model(self):
        """Load emotion ONNX model from local file."""
        if ONNX_MODEL_PATH is None or not Path(ONNX_MODEL_PATH).exists():
            logger.error(f"ONNX emotion model not found at: {ONNX_MODEL_PATH}")
            raise FileNotFoundError(f"ONNX emotion model not found: {ONNX_MODEL_PATH}")
        
        logger.info(f"Loading ONNX emotion model from: {ONNX_MODEL_PATH}")
        return cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)

    def predict_emotion(self, face_img):
        if face_img.size == 0: return "Unknown", 0.0

        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
            
        blob = cv2.dnn.blobFromImage(cv2.resize(gray, (64, 64)), 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
        self.emotion_net.setInput(blob)
        preds = self.emotion_net.forward()[0]
        
        preds = np.exp(preds - np.max(preds))
        probs = preds / np.sum(preds)
        best_idx = np.argmax(probs)
        return EMOTIONS_LABELS[best_idx], float(probs[best_idx])

    def check_zone(self, point, frame_shape):
        height, width = frame_shape[:2]
        polygons = self.settings.get("1", {}).get("polygons", [])
        
        for poly in polygons:
            points = np.array([
                [int(p["x"] * width), int(p["y"] * height)] 
                for p in poly.get("normalized", [])
            ], dtype=np.int32)
            
            if cv2.pointPolygonTest(points, point, False) >= 0:
                return True
        return False

    def clean_old_ids(self, current_active_ids):
        """
        Clean up buffers for people who left the screen.
        """
        # 1. Clean up the voting buffer
        buffer_ids = list(self.track_emotion_buffer.keys())
        for pid in buffer_ids:
            if pid not in current_active_ids:
                del self.track_emotion_buffer[pid]
        
    def _get_detections_from_service(self, frame) -> np.ndarray:
        """
        Send frame to remote YOLO service and return detections in format expected by tracker.
        Returns: np.ndarray of shape (N, 6) -> [x1, y1, x2, y2, conf, cls]
        """
        try:
            # Compress frame to JPEG to send
            _, img_encoded = cv2.imencode(".jpg", frame)
            files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

            resp = requests.post(self.service_url, files=files, timeout=5)
            resp.raise_for_status()

            # Parse response
            result = resp.json()
            predict_list = result.get("predict", [])

            detections = []
            if predict_list:
                data = json.loads(predict_list[0])
                for item in data:
                    cls_ = int(item.get("class", -1))
                    # Check if class is person (class 0) - Adjust based on your model!
                    # If this is the plate/customer model, check if 'name' is 'person' or class is 0
                    # Assuming class 0 is person for standard YOLO
                    if cls_ != 0:
                        continue 

                    box = item.get("box", {})
                    conf = item.get("confidence", 0.0)
                    x1, y1, x2, y2 = (
                        box.get("x1"),
                        box.get("y1"),
                        box.get("x2"),
                        box.get("y2"),
                    )

                    if x1 is not None:
                        detections.append([x1, y1, x2, y2, conf, cls_])

            if not detections:
                return np.empty((0, 6))

            return np.array(detections)

        except Exception as e:
            logger.error(f"Error calling YOLO service: {e}")
            return np.empty((0, 6))

    def process_frame(self, frame, timestamp):
        h, w = frame.shape[:2]
        
        # 1. Remote Detection
        det_array = self._get_detections_from_service(frame)
        
        # 2. Local Tracking
        if self.tracker is None:
            return

        frame_height, frame_width = frame.shape[:2]
        orig_shape = (frame_height, frame_width)
        mock_results = MockResults(det_array, orig_shape)
        
        # Update tracker
        tracks = self.tracker.update(mock_results, frame)
        # tracks: (N, 7) -> [x1, y1, x2, y2, id, conf, cls] or similar
        
        boxes = []
        track_ids = []
        
        for track in tracks:
            # Unwrap track info
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            
            boxes.append([x1, y1, x2, y2])
            track_ids.append(track_id)

        # Clean up old data
        self.clean_old_ids(set(track_ids))

        emotion_events = []

        for box, track_id in zip(boxes, track_ids):
            current_id = int(track_id)

            # 1. SKIP IF DONE: If this person is already finalized, skip
            if current_id in self.processed_track_ids:
                continue

            x1, y1, x2, y2 = box
            feet_point = (int((x1 + x2) / 2), int(y2))
            
            if self.check_zone(feet_point, frame.shape):
                safe_y1, safe_y2 = max(0, y1), min(h, y2)
                safe_x1, safe_x2 = max(0, x1), min(w, x2)
                body_roi = frame[safe_y1:safe_y2, safe_x1:safe_x2]
                
                if body_roi.size > 0:
                    rgb_body = cv2.cvtColor(body_roi, cv2.COLOR_BGR2RGB)
                    
                    try:
                        faces = self.mtcnn_detector.detect_faces(rgb_body)
                    except: faces = []
                    
                    if faces:
                        best_face = max(faces, key=lambda x: x['confidence'])
                        fx, fy, fw, fh = best_face['box']
                        face_img = body_roi[max(0,fy):fy+fh, max(0,fx):fx+fw]
                        
                        emotion, score = self.predict_emotion(face_img)
                        
                        if score > CONFIDENCE_THRESHOLD:
                            # 2. ADD TO BUFFER: Don't push yet, just store it
                            if current_id not in self.track_emotion_buffer:
                                self.track_emotion_buffer[current_id] = []
                            
                            self.track_emotion_buffer[current_id].append({
                                "emotion": emotion,
                                "score": score,
                                "bbox": [x1, y1, x2, y2]
                            })
                            
                            # 3. CHECK BUFFER SIZE: Do we have enough votes?
                            if len(self.track_emotion_buffer[current_id]) >= VOTE_COUNT:
                                # Calculate Mode (Most Frequent Emotion)
                                emotions_list = [e['emotion'] for e in self.track_emotion_buffer[current_id]]
                                
                                # --- HAPPY PRIORITY LOGIC ---
                                # If 'Happy' appeared even once, pick it. Otherwise, use majority.
                                if 'Happy' in emotions_list:
                                    final_emotion = 'Happy'
                                else:
                                    # Fallback to majority vote
                                    final_emotion = Counter(emotions_list).most_common(1)[0][0]
                                
                                # Find the best score for that specific emotion
                                best_entry = max(
                                    [e for e in self.track_emotion_buffer[current_id] if e['emotion'] == final_emotion],
                                    key=lambda x: x['score']
                                )
                                
                                # 4. PUSH AND LOCK
                                self.processed_track_ids.add(current_id) # Lock ID
                                del self.track_emotion_buffer[current_id] # Clear Buffer
                                
                                logger.info(f"Final Decision for ID {current_id}: {final_emotion}")
                                
                                emotion_events.append({
                                    "track_id": current_id,
                                    "emotion": final_emotion,
                                    "score": round(best_entry['score'], 2),
                                    "bbox": best_entry['bbox']
                                })

        if emotion_events:
            self.send_alert(frame, emotion_events, timestamp)

    def send_alert(self, frame, events, timestamp):
        try:
            alert_id = str(uuid.uuid4())
            
            # Check if any detected emotion is 'Happy'
            happy_events = [e for e in events if e['emotion'] == 'Happy']
            
            # 1. Base payload (Always goes to main stats)
            main_payload = {
                "camera_id": CAMERA_ID,
                "timestamp": timestamp,
                "emotions": events,
                "description": f"Detected {len(events)} exiting people (Mode).",
                "target_collection": TARGET_COLLECTION,
                "ticket_id": alert_id
            }
            # Push to main stats immediately
            self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(main_payload)})
            
            # 2. Alert Payload (Only if Happy)
            if happy_events:
                # We create a separate payload for the alert
                alert_payload = main_payload.copy()
                
                # Encode RAW frame first (without annotations)
                _, raw_buffer = cv2.imencode('.jpg', frame)
                raw_frame_b64 = base64.b64encode(raw_buffer).decode('utf-8')

                annotated_frame = frame.copy()
                for event in events:
                    x1, y1, x2, y2 = event['bbox']
                    emotion = event['emotion']
                    score = event['score']
                    
                    # Highlight Happy faces specifically in Green, others in White
                    color = (0, 255, 0) if emotion == 'Happy' else (255, 255, 255)
                    thickness = 3 if emotion == 'Happy' else 2
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"{emotion} ({score:.2f})"
                    if emotion == 'Happy':
                        label = "ALERTA: HAPPY FACE! " + label
                        
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Convert annotated frame to base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Special alert payload properties
                alert_payload["ticket_id"] = alert_id
                alert_payload["frame"] = frame_b64
                alert_payload["frame_without_visualization"] = raw_frame_b64
#                alert_payload["description"] = f"Happy face detected! Total exiting: {len(events)}"
                alert_payload["description"] = f"Happy person detected."                # 1. RAISE TICKET
                alert_payload["type"] = "raised"
                alert_payload["target_collection"] = "alert_collection" 
                
                logger.info(f"WOW! Happy face detected. Raising Alert ID: {alert_id}")
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(alert_payload)})

                # 2. INSTANTLY RESOLVE TICKET
                # We reuse the same payload but update type and description for resolution
                resolve_payload = alert_payload.copy()
                resolve_payload["type"] = "resolved"
                resolve_payload["description"] = f"Happy face verified. Auto-resolved: {alert_id}"
                
                logger.info(f"Auto-resolving Alert ID: {alert_id}")
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(resolve_payload)})

            logger.info(f"Event Pushed. Total people: {len(events)}. Happy: {len(happy_events) > 0}")
            
        except Exception as e:
            logger.error(f"Failed to push alert: {e}")

    def draw_visualization(self, frame):
        """Draw detection zones, bounding boxes, track IDs, and emotions on the frame."""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Draw detection zones
        polygons = self.settings.get("1", {}).get("polygons", [])
        for poly in polygons:
            points = np.array([
                [int(p["x"] * w), int(p["y"] * h)] 
                for p in poly.get("normalized", [])
            ], dtype=np.int32)
            
            # Draw polygon outline
            cv2.polylines(vis_frame, [points], True, (0, 255, 255), 2)
            
            # Draw semi-transparent fill
            overlay = vis_frame.copy()
            cv2.fillPoly(overlay, [points], (0, 255, 255))
            cv2.addWeighted(overlay, 0.2, vis_frame, 0.8, 0, vis_frame)
        
        # Draw current tracks from buffer
        for track_id, emotions in self.track_emotion_buffer.items():
            if emotions:
                latest = emotions[-1]
                bbox = latest.get("bbox", [0, 0, 0, 0])
                emotion = latest.get("emotion", "Unknown")
                score = latest.get("score", 0)
                
                x1, y1, x2, y2 = bbox
                
                # Color based on emotion
                emotion_colors = {
                    'Happy': (0, 255, 0),
                    'Sad': (255, 0, 0),
                    'Anger': (0, 0, 255),
                    'Neutral': (200, 200, 200),
                    'Surprise': (255, 255, 0),
                    'Fear': (255, 0, 255),
                    'Disgust': (0, 100, 0),
                    'Contempt': (100, 100, 100),
                }
                color = emotion_colors.get(emotion, (255, 255, 255))
                thickness = 2
                
                # Make Happy more prominent in visualization
                if emotion == 'Happy':
                    thickness = 3
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw track ID and emotion
                label = f"ID:{track_id} {emotion} ({score:.2f})"
                if emotion == 'Happy':
                    label = "HAPPY! " + label
                cv2.putText(vis_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                
                # Draw vote progress
                votes = len(emotions)
                vote_text = f"Votes: {votes}/{VOTE_COUNT}"
                cv2.putText(vis_frame, vote_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw status
        cv2.putText(vis_frame, f"Active: {len(self.track_emotion_buffer)} | Processed: {len(self.processed_track_ids)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame

    def run(self):
        logger.info(f"Starting Service. Target Camera: {CAMERA_SOURCE}")
        src = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
        
        while True:
            try:
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    logger.warning(f"Unable to open camera: {src}. Retrying...")
                    time.sleep(5)
                    continue
                
                logger.info(f"Camera connected: {src}")
                source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_interval = max(1, int(source_fps / TARGET_FPS))
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        try:
                            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                            self.process_frame(frame, timestamp)
                            
                            # Push visualization frame if enabled
                            if VISUALIZE and hasattr(self, 'vis_queue'):
                                try:
                                    # Draw visualization on frame
                                    vis_frame = self.draw_visualization(frame)
                                    
                                    _, img_encoded = cv2.imencode('.jpg', vis_frame)
                                    frame_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
                                    
                                    # Get current buffer states for display
                                    buffer_info = []
                                    for tid, emotions in self.track_emotion_buffer.items():
                                        buffer_info.append({
                                            "track_id": tid,
                                            "vote_count": len(emotions),
                                            "latest_emotion": emotions[-1]["emotion"] if emotions else "Unknown",
                                        })
                                    
                                    vis_payload = {
                                        "camera_id": CAMERA_ID,
                                        "timestamp": timestamp,
                                        "frame": frame_b64,
                                        "frame_num": frame_count,
                                        "active_tracks": len(self.track_emotion_buffer),
                                        "processed_count": len(self.processed_track_ids),
                                        "buffer_info": buffer_info,
                                    }
                                    self.vis_queue.push(json.dumps(vis_payload))
                                except Exception as e:
                                    logger.error(f"Failed to push visualization frame: {e}")
                                    
                        except Exception as e:
                            logger.error(f"Processing error: {e}")

                    frame_count += 1
                
                cap.release()
                
            except Exception as e:
                logger.error(f"Video loop error: {e}. Retrying...")
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                time.sleep(5)

if __name__ == "__main__":
    service = DirectEmotionService()
    service.run()




# import json
# import numpy as np
# import cv2
# from dotenv import load_dotenv
# from pathlib import Path
# import redis
# import logging
# import time
# import requests
# import logging_loki
# from multiprocessing import Queue
# import os
# import base64
# from collections import deque, Counter
# import uuid
# import urllib.request

# # AI Models
# from ultralytics import YOLO
# from mtcnn import MTCNN

# load_dotenv()

# # --- CONFIGURATION ---
# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
# LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

# # Stream Config
# CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
# TARGET_FPS = int(os.getenv("TARGET_FPS", "5"))

# # Model paths - use local files from service directory
# SERVICE_DIR = Path(__file__).parent.resolve()
# WORKSPACE_DIR = Path("/workspace")

# # Check workspace first (Docker), then service directory (local dev)
# def get_model_path(filename, env_var=None, default=None):
#     """Find model file in workspace or service directory."""
#     workspace_path = WORKSPACE_DIR / filename
#     service_path = SERVICE_DIR / filename
    
#     if workspace_path.exists():
#         return str(workspace_path)
#     elif service_path.exists():
#         return str(service_path)
#     elif env_var:
#         return os.getenv(env_var, default)
#     return None

# YOLO_MODEL_PATH = get_model_path("yolo11n.pt", "YOLO_MODEL_PATH", "yolov8n.pt")
# ONNX_MODEL_PATH = get_model_path("emotion-ferplus-8.onnx")


# # Output
# OUTPUT_QUEUE_1 = os.getenv("OUTPUT_QUEUE_NAME_1", "emotion_alerts")
# TARGET_COLLECTION = os.getenv("TARGET_COLLECTION", "emotion_events")
# CAMERA_ID = os.getenv("CAMERA_ID", "smilometer")
# VISUALIZE = str(os.getenv("VISUALIZE", "False")).lower() == "true"

# # Constants
# CONFIDENCE_THRESHOLD = 0.6
# EMOTIONS_LABELS = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']

# # VOTING CONFIG
# VOTE_COUNT = 3  # Number of frames to collect before deciding the emotion

# # --- LOGGING SETUP ---
# try:
#     handler = logging_loki.LokiQueueHandler(
#         Queue(-1),
#         url=LOKI_URL,
#         tags={"application": "direct_emotion_service"},
#         version="1",
#     )
#     logger = logging.getLogger("direct_emotion_service")
#     logger.addHandler(handler)
# except Exception:
#     logger = logging.getLogger("direct_emotion_service")

# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())


# class FixedSizeQueue:
#     def __init__(self, redis_client, name, max_size):
#         self.r = redis_client
#         self.name = name
#         self.max_size = max_size

#     def push(self, item):
#         """
#         Push item to queue.
#         Oldest items are dropped.
#         """
#         pipe = self.r.pipeline()
#         pipe.rpush(self.name, item)
#         pipe.ltrim(self.name, -self.max_size, -1)
#         pipe.execute()

#     def pop(self, timeout=0):
#         """
#         Blocking FIFO pop.
#         """
#         result = self.r.blpop(self.name, timeout=timeout)
#         if result:
#             _, value = result
#             return value
#         return None

#     def size(self):
#         return self.r.llen(self.name)


# class DirectEmotionService:
#     def __init__(self):
#         self.settings = self.load_settings()
#         self.redis = self.connect_redis()
        
#         # 1. COMPLETED: IDs that have already been pushed to Redis
#         self.processed_track_ids = set()
        
#         # 2. BUFFER: Temporary storage for voting { track_id: [emotion1, emotion2, ...] }
#         self.track_emotion_buffer = {}
        
#         logger.info(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
#         if not Path(YOLO_MODEL_PATH).exists():
#             logger.error(f"YOLO model not found at: {YOLO_MODEL_PATH}")
#             raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")
#         self.tracking_model = YOLO(YOLO_MODEL_PATH)

#         self.mtcnn_detector = MTCNN()
#         self.emotion_net = self.load_onnx_model()
        
#         # Setup visualization queue if enabled
#         if VISUALIZE:
#             self.vis_queue = FixedSizeQueue(
#                 self.redis, f"stream:visualization:{CAMERA_ID}", 10
#             )
#             logger.info(f"Visualization queue initialized: stream:visualization:{CAMERA_ID}")
        
#     def load_settings(self):
#         base_dir = Path(__file__).parent.resolve()
#         settings_path = base_dir / "setting.json"
#         try:
#             with open(settings_path, "r") as f:
#                 return json.load(f)
#         except Exception as e:
#             logger.error(f"Error loading settings: {e}")
#             return {}

#     def connect_redis(self):
#         while True:
#             try:
#                 r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
#                 r.ping()
#                 logger.info("Connected to Redis output.")
#                 return r
#             except Exception as e:
#                 logger.error(f"Redis failed: {e}. Retrying in 5s...")
#                 time.sleep(5)

#     def load_onnx_model(self):
#         """Load emotion ONNX model from local file."""
#         if ONNX_MODEL_PATH is None or not Path(ONNX_MODEL_PATH).exists():
#             logger.error(f"ONNX emotion model not found at: {ONNX_MODEL_PATH}")
#             raise FileNotFoundError(f"ONNX emotion model not found: {ONNX_MODEL_PATH}")
        
#         logger.info(f"Loading ONNX emotion model from: {ONNX_MODEL_PATH}")
#         return cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)

#     def predict_emotion(self, face_img):
#         if face_img.size == 0: return "Unknown", 0.0

#         if len(face_img.shape) == 3:
#             gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = face_img
            
#         blob = cv2.dnn.blobFromImage(cv2.resize(gray, (64, 64)), 1.0/255.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)        self.emotion_net.setInput(blob)
#         preds = self.emotion_net.forward()[0]
        
#         preds = np.exp(preds - np.max(preds))
#         probs = preds / np.sum(preds)
#         best_idx = np.argmax(probs)
#         return EMOTIONS_LABELS[best_idx], float(probs[best_idx])

#     def check_zone(self, point, frame_shape):
#         height, width = frame_shape[:2]
#         polygons = self.settings.get("1", {}).get("polygons", [])
        
#         for poly in polygons:
#             points = np.array([
#                 [int(p["x"] * width), int(p["y"] * height)] 
#                 for p in poly.get("normalized", [])
#             ], dtype=np.int32)
            
#             if cv2.pointPolygonTest(points, point, False) >= 0:
#                 return True
#         return False

#     def clean_old_ids(self, current_active_ids):
#         """
#         Clean up buffers for people who left the screen.
#         """
#         # 1. Clean up the voting buffer
#         buffer_ids = list(self.track_emotion_buffer.keys())
#         for pid in buffer_ids:
#             if pid not in current_active_ids:
#                 del self.track_emotion_buffer[pid]
        
#         # 2. Optional: Allow re-entry by clearing processed_ids? 
#         # For now, we keep them in processed_track_ids so they are alerted only ONCE per session.
#         # If you want them to be alerted again if they leave and come back, uncomment below:
#         # known_ids = list(self.processed_track_ids)
#         # for pid in known_ids:
#         #    if pid not in current_active_ids:
#         #        self.processed_track_ids.remove(pid)

#     def process_frame(self, frame, timestamp):
#         h, w = frame.shape[:2]
        
#         results = self.tracking_model.track(frame, persist=True, verbose=False, classes=0)
        
#         if not results or results[0].boxes.id is None:
#             return

#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         track_ids = results[0].boxes.id.int().cpu().numpy().tolist()

#         # Clean up old data
#         self.clean_old_ids(set(track_ids))

#         emotion_events = []

#         for box, track_id in zip(boxes, track_ids):
#             current_id = int(track_id)

#             # 1. SKIP IF DONE: If this person is already finalized, skip
#             if current_id in self.processed_track_ids:
#                 continue

#             x1, y1, x2, y2 = map(int, box)
#             feet_point = (int((x1 + x2) / 2), int(y2))
            
#             if self.check_zone(feet_point, frame.shape):
#                 safe_y1, safe_y2 = max(0, y1), min(h, y2)
#                 safe_x1, safe_x2 = max(0, x1), min(w, x2)
#                 body_roi = frame[safe_y1:safe_y2, safe_x1:safe_x2]
                
#                 if body_roi.size > 0:
#                     rgb_body = cv2.cvtColor(body_roi, cv2.COLOR_BGR2RGB)
                    
#                     try:
#                         faces = self.mtcnn_detector.detect_faces(rgb_body)
#                     except: faces = []
                    
#                     if faces:
#                         best_face = max(faces, key=lambda x: x['confidence'])
#                         fx, fy, fw, fh = best_face['box']
#                         face_img = body_roi[max(0,fy):fy+fh, max(0,fx):fx+fw]
                        
#                         emotion, score = self.predict_emotion(face_img)
                        
#                         if score > CONFIDENCE_THRESHOLD:
#                             # 2. ADD TO BUFFER: Don't push yet, just store it
#                             if current_id not in self.track_emotion_buffer:
#                                 self.track_emotion_buffer[current_id] = []
                            
#                             self.track_emotion_buffer[current_id].append({
#                                 "emotion": emotion,
#                                 "score": score,
#                                 "bbox": [x1, y1, x2, y2]
#                             })
                            
#                             # 3. CHECK BUFFER SIZE: Do we have enough votes?
#                             # if len(self.track_emotion_buffer[current_id]) >= VOTE_COUNT:
#                             #     # Calculate Mode (Most Frequent Emotion)
#                             #     emotions_list = [e['emotion'] for e in self.track_emotion_buffer[current_id]]
#                             #     most_common_emotion = Counter(emotions_list).most_common(1)[0][0]
                                
#                             #     # Find the best score for that specific emotion
#                             #     best_entry = max(
#                             #         [e for e in self.track_emotion_buffer[current_id] if e['emotion'] == most_common_emotion],
#                             #         key=lambda x: x['score']
#                             #     )

#                             if len(self.track_emotion_buffer[current_id]) >= VOTE_COUNT:
#                                 emotions_list = [e['emotion'] for e in self.track_emotion_buffer[current_id]]
                                
#                                 # --- NEW PRIORITY LOGIC ---
#                                 # If 'Happy' appeared even once, pick it. Otherwise, use majority.
#                                 if 'Happy' in emotions_list:
#                                     final_emotion = 'Happy'
#                                 else:
#                                     # Fallback to majority vote
#                                     final_emotion = Counter(emotions_list).most_common(1)[0][0]
                                
#                                 # Find the best entry matching our decision
#                                 # (Use a default filter to prevent crashes if the list logic mismatches)
#                                 matching_entries = [e for e in self.track_emotion_buffer[current_id] if e['emotion'] == final_emotion]
#                                 if matching_entries:
#                                     best_entry = max(matching_entries, key=lambda x: x['score'])
#                                 else:
#                                     # Fallback if something weird happens
#                                     best_entry = self.track_emotion_buffer[current_id][-1]
                                
#                                 # 4. PUSH AND LOCK
#                                 self.processed_track_ids.add(current_id) # Lock ID
#                                 del self.track_emotion_buffer[current_id] # Clear Buffer
                                
#                                 logger.info(f"Final Decision for ID {current_id}: {most_common_emotion}")
                                
#                                 emotion_events.append({
#                                     "track_id": current_id,
#                                     "emotion": most_common_emotion,
#                                     "score": round(best_entry['score'], 2),
#                                     "bbox": best_entry['bbox']
#                                 })

#         if emotion_events:
#             self.send_alert(frame, emotion_events, timestamp)

#     # def send_alert(self, frame, events, timestamp):
#     #     try:
#     #         alert_id = str(uuid.uuid4())
            
#     #         # Check if any detected emotion is 'Happy'
#     #         happy_events = [e for e in events if e['emotion'] == 'Happy']
            
#     #         # Base payload for all emotion events
#     #         payload = {
#     #             "camera_id": CAMERA_ID,
#     #             "timestamp": timestamp,
#     #             "emotions": events,
#     #             "description": f"Detected {len(events)} exiting people (Mode).",
#     #             "target_collection": TARGET_COLLECTION,
#     #             "ticket_id": alert_id
#     #         }
            
#     #         # If happy faces are found, we raise a special alert with the annotated frame
#     #         if happy_events:
#     #             annotated_frame = frame.copy()
#     #             for event in events:
#     #                 x1, y1, x2, y2 = event['bbox']
#     #                 emotion = event['emotion']
#     #                 score = event['score']
                    
#     #                 # Highlight Happy faces specifically in Green, others in White
#     #                 color = (0, 255, 0) if emotion == 'Happy' else (255, 255, 255)
#     #                 thickness = 3 if emotion == 'Happy' else 2
                    
#     #                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
#     #                 label = f"{emotion} ({score:.2f})"
#     #                 if emotion == 'Happy':
#     #                     label = "ALERTA: HAPPY FACE! " + label
                        
#     #                 cv2.putText(annotated_frame, label, (x1, y1 - 10), 
#     #                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     #             # Convert to base64 for storage/alert
#     #             _, buffer = cv2.imencode('.jpg', annotated_frame)
#     #             frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
#     #             # Special alert payload properties
#     #             payload["frame"] = frame_b64
#     #             payload["description"] = f"Happy face detected! Total exiting: {len(events)}"
#     #             payload["type"] = "happy_face_alert"
#     #             # You might want to store alerts in a specific collection
#     #             payload["target_collection"] = "alert_collection" 
                
#     #             logger.info(f"WOW! Happy face detected. Alert ID: {alert_id}")
#             #     logger.info(f"Event Pushed. Total people: {len(events)}. Happy: {len(happy_events) > 0}")

#     #         self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(payload)})
#     def send_alert(self, frame, events, timestamp):
#         try:
#             alert_id = str(uuid.uuid4())
#             happy_events = [e for e in events if e['emotion'] == 'Happy']
            
#             # 1. Standard Payload (Always goes to main stats)
#             main_payload = {
#                 "camera_id": CAMERA_ID,
#                 "timestamp": timestamp,
#                 "emotions": events,
#                 "description": f"Detected {len(events)} exiting people.",
#                 "target_collection": TARGET_COLLECTION, # e.g., 'emotion_events'
#                 "ticket_id": alert_id
#             }
#             # Push to main stats immediately
#             self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(main_payload)})

#             # 2. Alert Payload (Only if Happy)
#             if happy_events:
#                 annotated_frame = frame.copy()
#                 for event in events:
#                     # ... drawing logic ...
#                     x1, y1, x2, y2 = event['bbox']
#                     emotion = event['emotion']
#                     score = event['score']
#                     color = (0, 255, 0) if emotion == 'Happy' else (255, 255, 255)
#                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
#                     # ... label logic ...

#                 _, buffer = cv2.imencode('.jpg', annotated_frame)
#                 frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
#                 alert_payload = {
#                     "camera_id": CAMERA_ID,
#                     "timestamp": timestamp,
#                     "emotions": happy_events, # Only send the happy data in the alert? Or all?
#                     "frame": frame_b64,
#                     "description": "Happy face detected!",
#                     "type": "happy_face_alert",
#                     "target_collection": "alert_collection",
#                     "ticket_id": alert_id
#                 }
#                 # Push separate alert message
#                 self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(alert_payload)})
#                 logger.info(f"Happy Alert Pushed separately for ID: {alert_id}")


#         except Exception as e:
#             logger.error(f"Failed to push alert: {e}")

#     def draw_visualization(self, frame):
#         """Draw detection zones, bounding boxes, track IDs, and emotions on the frame."""
#         vis_frame = frame.copy()
#         h, w = vis_frame.shape[:2]
        
#         # Draw detection zones
#         polygons = self.settings.get("1", {}).get("polygons", [])
#         for poly in polygons:
#             points = np.array([
#                 [int(p["x"] * w), int(p["y"] * h)] 
#                 for p in poly.get("normalized", [])
#             ], dtype=np.int32)
            
#             # Draw polygon outline
#             cv2.polylines(vis_frame, [points], True, (0, 255, 255), 2)
            
#             # Draw semi-transparent fill
#             overlay = vis_frame.copy()
#             cv2.fillPoly(overlay, [points], (0, 255, 255))
#             cv2.addWeighted(overlay, 0.2, vis_frame, 0.8, 0, vis_frame)
        
#         # Draw current tracks from buffer
#         for track_id, emotions in self.track_emotion_buffer.items():
#             if emotions:
#                 latest = emotions[-1]
#                 bbox = latest.get("bbox", [0, 0, 0, 0])
#                 emotion = latest.get("emotion", "Unknown")
#                 score = latest.get("score", 0)
                
#                 x1, y1, x2, y2 = bbox
                
#                 # Color based on emotion
#                 emotion_colors = {
#                     'Happy': (0, 255, 0),
#                     'Sad': (255, 0, 0),
#                     'Anger': (0, 0, 255),
#                     'Neutral': (200, 200, 200),
#                     'Surprise': (255, 255, 0),
#                     'Fear': (255, 0, 255),
#                     'Disgust': (0, 100, 0),
#                     'Contempt': (100, 100, 100),
#                 }
#                 color = emotion_colors.get(emotion, (255, 255, 255))
#                 thickness = 2
                
#                 # Make Happy more prominent in visualization
#                 if emotion == 'Happy':
#                     thickness = 3
                
#                 # Draw bounding box
#                 cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
                
#                 # Draw track ID and emotion
#                 label = f"ID:{track_id} {emotion} ({score:.2f})"
#                 if emotion == 'Happy':
#                     label = "HAPPY! " + label
#                 cv2.putText(vis_frame, label, (x1, y1 - 10), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                
#                 # Draw vote progress
#                 votes = len(emotions)
#                 vote_text = f"Votes: {votes}/{VOTE_COUNT}"
#                 cv2.putText(vis_frame, vote_text, (x1, y2 + 20), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
#         # Draw status
#         cv2.putText(vis_frame, f"Active: {len(self.track_emotion_buffer)} | Processed: {len(self.processed_track_ids)}", 
#                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         return vis_frame

#     def run(self):
#         logger.info(f"Starting Service. Target Camera: {CAMERA_SOURCE}")
#         src = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
        
#         while True:
#             try:
#                 cap = cv2.VideoCapture(src)
#                 if not cap.isOpened():
#                     logger.warning(f"Unable to open camera: {src}. Retrying...")
#                     time.sleep(5)
#                     continue
                
#                 logger.info(f"Camera connected: {src}")
#                 source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
#                 frame_interval = max(1, int(source_fps / TARGET_FPS))
#                 frame_count = 0

#                 while True:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break

#                     if frame_count % frame_interval == 0:
#                         try:
#                             timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#                             self.process_frame(frame, timestamp)
                            
#                             # Push visualization frame if enabled
#                             if VISUALIZE and hasattr(self, 'vis_queue'):
#                                 try:
#                                     # Draw visualization on frame
#                                     vis_frame = self.draw_visualization(frame)
                                    
#                                     _, img_encoded = cv2.imencode('.jpg', vis_frame)
#                                     frame_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
                                    
#                                     # Get current buffer states for display
#                                     buffer_info = []
#                                     for tid, emotions in self.track_emotion_buffer.items():
#                                         buffer_info.append({
#                                             "track_id": tid,
#                                             "vote_count": len(emotions),
#                                             "latest_emotion": emotions[-1]["emotion"] if emotions else "Unknown",
#                                         })
                                    
#                                     vis_payload = {
#                                         "camera_id": CAMERA_ID,
#                                         "timestamp": timestamp,
#                                         "frame": frame_b64,
#                                         "frame_num": frame_count,
#                                         "active_tracks": len(self.track_emotion_buffer),
#                                         "processed_count": len(self.processed_track_ids),
#                                         "buffer_info": buffer_info,
#                                     }
#                                     self.vis_queue.push(json.dumps(vis_payload))
#                                 except Exception as e:
#                                     logger.error(f"Failed to push visualization frame: {e}")
                                    
#                         except Exception as e:
#                             logger.error(f"Processing error: {e}")

#                     frame_count += 1
                
#                 cap.release()
                
#             except Exception as e:
#                 logger.error(f"Video loop error: {e}. Retrying...")
#                 if 'cap' in locals() and cap.isOpened():
#                     cap.release()
#                 time.sleep(5)

# if __name__ == "__main__":
#     service = DirectEmotionService()
#     service.run()