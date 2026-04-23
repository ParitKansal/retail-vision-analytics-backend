# # import json
# # import numpy as np
# # import cv2
# # from dotenv import load_dotenv
# # from pathlib import Path
# # import redis
# # import logging
# # import time
# # import requests
# # import logging_loki
# # from multiprocessing import Queue
# # import os
# # import base64
# # from collections import deque, Counter
# # import uuid
# # import torch
# # from torchvision import transforms
# # from PIL import Image
# # import matplotlib.pyplot as plt
# # from ultralytics import YOLO
# # from transformers import ViTForImageClassification, BitsAndBytesConfig
# # import bitsandbytes

# # # AI Models
# # from ultralytics import YOLO

# # load_dotenv()

# # # --- CONFIGURATION ---
# # REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# # REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
# # LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

# # # Stream Config
# # CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "6")
# # TARGET_FPS = int(os.getenv("TARGET_FPS", "1"))

# # # Model paths - use local files from service directory
# # SERVICE_DIR = Path(__file__).parent.resolve()
# # WORKSPACE_DIR = Path("/workspace")
# # def get_model_path(filename, env_var=None, default=None):
# #     """Find model file in workspace or service directory."""
# #     workspace_path = WORKSPACE_DIR / filename
# #     service_path = SERVICE_DIR / filename
    
# #     if workspace_path.exists():
# #         return str(workspace_path)
# #     elif service_path.exists():
# #         return str(service_path)
# #     elif env_var:
# #         return os.getenv(env_var, default)
# #     return None

# # YOLO_MODEL = os.getenv("YOLO_MODEL", "face_detection_06022026.pt")
# # BNB_4BIT_PATH = os.getenv("BNB_4BIT_PATH", "bnb_4bit_model")

# # # Output
# # OUTPUT_QUEUE_1 = os.getenv("OUTPUT_QUEUE_NAME_1", "emotion_alerts")
# # TARGET_COLLECTION = os.getenv("TARGET_COLLECTION", "emotion_events")
# # CAMERA_ID = os.getenv("CAMERA_ID", "smilometer")
# # VISUALIZE = str(os.getenv("VISUALIZE", "False")).lower() == "true"

# # # Constants
# # # CONFIDENCE_THRESHOLD = 0.6
# # EMOTIONS_LABELS = ['neutral', 'happy']

# # # VOTING CONFIG
# # VOTE_COUNT = 3  # Number of frames to collect before deciding the emotion

# # # --- LOGGING SETUP ---
# # try:
# #     handler = logging_loki.LokiQueueHandler(
# #         Queue(-1),
# #         url=LOKI_URL,
# #         tags={"application": "direct_emotion_service"},
# #         version="1",
# #     )
# #     logger = logging.getLogger("direct_emotion_service")
# #     logger.addHandler(handler)
# # except Exception:
# #     logger = logging.getLogger("direct_emotion_service")

# # logger.setLevel(logging.INFO)
# # logger.addHandler(logging.StreamHandler())

# # class FixedSizeQueue:
# #     def __init__(self, redis_client, name, max_size):
# #         self.r = redis_client
# #         self.name = name
# #         self.max_size = max_size

# #     def push(self, item):
# #         """
# #         Push item to queue.
# #         Oldest items are dropped.
# #         """
# #         pipe = self.r.pipeline()
# #         pipe.rpush(self.name, item)
# #         pipe.ltrim(self.name, -self.max_size, -1)
# #         pipe.execute()

# #     def pop(self, timeout=0):
# #         """
# #         Blocking FIFO pop.
# #         """
# #         result = self.r.blpop(self.name, timeout=timeout)
# #         if result:
# #             _, value = result
# #             return value
# #         return None

# #     def size(self):
# #         return self.r.llen(self.name)

# # class Config:
# #     FRAME_INTERVAL = 1  # Process 1 frame per second
# #     FACE_CONFIDENCE_THRESHOLD = 0.4
# #     PADDING_PERCENT = 0.3 
# #     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# #     BNB_4BIT_PATH = os.getenv("BNB_4BIT_PATH", "bnb_4bit_model")

# #     # Model configuration
# #     ID2LABEL = {0: 'neutral', 1: 'happy'}
# #     LABEL2ID = {'neutral': 0, 'happy': 1}

# #     # Class names (must match training)
# #     CLASS_NAMES = ['Neutral', 'Happy']

# # class DirectEmotionService:
# #     def __init__(self):
# #         self.settings = self.load_settings()
# #         self.redis = self.connect_redis()
# #         self.results = []
# #         # 1. COMPLETED: IDs that have already been pushed to Redis
# #         self.processed_track_ids = set()
        
# #         # 2. BUFFER: Temporary storage for voting { track_id: [emotion1, emotion2, ...] }
# #         self.track_emotion_buffer = {}
        
# #         logger.info(f"Loading YOLO model")
# #         self.face_model = YOLO(YOLO_MODEL)
        
# #         logger.info(f"Loading Emotion model")
# #         self.emotion_model = self.load_models()
        
# #         # Transforms for ViT
# #         self.transform = transforms.Compose([
# #             transforms.Resize((224, 224)),
# #             transforms.ToTensor(),
# #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# #         ])

# #         # Setup visualization queue if enabled
# #         if VISUALIZE:
# #             self.vis_queue = FixedSizeQueue(
# #                 self.redis, f"stream:visualization:{CAMERA_ID}", 10
# #             )
# #             logger.info(f"Visualization queue initialized: stream:visualization:{CAMERA_ID}")

# #     def load_settings(self):
# #         base_dir = Path(__file__).parent.resolve()
# #         settings_path = base_dir / "setting.json"
# #         try:
# #             with open(settings_path, "r") as f:
# #                 return json.load(f)
# #         except Exception as e:
# #             logger.error(f"Error loading settings: {e}")
# #             return {}

# #     def connect_redis(self):
# #         while True:
# #             try:
# #                 r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
# #                 r.ping()
# #                 logger.info("Connected to Redis output.")
# #                 return r
# #             except Exception as e:
# #                 logger.error(f"Redis failed: {e}. Retrying in 5s...")
# #                 time.sleep(5)
    
# #     def mask_frame(self, frame):
# #         height, width = frame.shape[:2]
# #         polygons = self.settings.get("1", {}).get("polygons", [])
# #         mask = np.zeros((height, width), dtype=np.uint8)
        
# #         for poly in polygons:
# #             points = np.array([
# #                 [int(p["x"] * width), int(p["y"] * height)] 
# #                 for p in poly.get("normalized", [])
# #             ], dtype=np.int32)
# #             cv2.fillPoly(mask, [points], 255)

# #         # Apply mask → everything outside polygon becomes black
# #         result = cv2.bitwise_and(frame, frame, mask=mask)   
# #         return result

# #     def clean_old_ids(self, current_active_ids):
# #         """
# #         Clean up buffers for people who left the screen.
# #         """
# #         # 1. Clean up the voting buffer
# #         buffer_ids = list(self.track_emotion_buffer.keys())
# #         for pid in buffer_ids:
# #             if pid not in current_active_ids:
# #                 del self.track_emotion_buffer[pid]
    
# #     def load_models(self):
# #         # Configure 4-bit quantization
# #         quantization_config = BitsAndBytesConfig(
# #             load_in_4bit=True,
# #             bnb_4bit_compute_dtype=torch.float16,
# #             bnb_4bit_use_double_quant=True,
# #             bnb_4bit_quant_type="nf4"
# #         )
        
# #         # Load 4-bit quantized model
# #         try:
# #             bnb_4bit_model = ViTForImageClassification.from_pretrained(
# #                 Config.BNB_4BIT_PATH,
# #                 quantization_config=quantization_config,
# #                 device_map="auto"
# #             )
# #             return bnb_4bit_model
# #         except Exception as e:
# #             logger.error(f"Failed to load user model: {e}")
# #             return None

# #     # def process_frame(self, frame, timestamp):
# #     #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     #     frame_h, frame_w = frame.shape[:2]
        
# #     #     masked_frame = self.mask_frame(frame)  
        
# #     #     # Use tracking instead of plain detection
# #     #     results = self.face_model(masked_frame, verbose=False)
        
# #     #     current_active_ids = set()
# #     #     emotion_events = []

# #     #     if results and results[0].boxes.id is not None:
# #     #         boxes = results[0].boxes.xyxy.cpu().numpy()
# #     #         track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
# #     #         confs = results[0].boxes.conf.cpu().numpy()
            
# #     #         current_active_ids = set(track_ids)

# #     #         for box, track_id, conf in zip(boxes, track_ids, confs):
# #     #             current_id = int(track_id)
                
# #     #             # 1. SKIP IF DONE
# #     #             if current_id in self.processed_track_ids:
# #     #                 continue

# #     #             # Track faces below threshold
# #     #             if conf < Config.FACE_CONFIDENCE_THRESHOLD:
# #     #                 continue
                
# #     #             x1, y1, x2, y2 = map(int, box)
                
# #     #             # Apply padding
# #     #             pad_x = int((x2 - x1) * Config.PADDING_PERCENT)
# #     #             pad_y = int((y2 - y1) * Config.PADDING_PERCENT)
                
# #     #             x1_p = max(0, x1 - pad_x)
# #     #             y1_p = max(0, y1 - pad_y)
# #     #             x2_p = min(frame_w, x2 + pad_x)
# #     #             y2_p = min(frame_h, y2 + pad_y)
                
# #     #             # Crop face
# #     #             face_crop = frame_rgb[y1_p:y2_p, x1_p:x2_p]
# #     #             if face_crop.size == 0:
# #     #                 continue
                
# #     #             # Prepare input for Emotion Model
# #     #             face_pil = Image.fromarray(face_crop)
# #     #             face_tensor = self.transform(face_pil).unsqueeze(0).to(Config.DEVICE)
                
# #     #             # ===== 4-BIT QUANTIZED MODEL INFERENCE =====
# #     #             if self.emotion_model is not None:
# #     #                 with torch.no_grad():
# #     #                     outputs = self.emotion_model(face_tensor).logits
# #     #                     probs = torch.softmax(outputs, dim=1)
# #     #                     confidence, predicted = torch.max(probs, 1)
                        
# #     #                     emotion = Config.CLASS_NAMES[predicted.item()]
# #     #                     score = confidence.item()
# #     #             else:
# #     #                 logger.error("4-bit model not loaded!")
# #     #                 continue
                
# #     #             # 2. ADD TO BUFFER
# #     #             if current_id not in self.track_emotion_buffer:
# #     #                 self.track_emotion_buffer[current_id] = []
                
# #     #             self.track_emotion_buffer[current_id].append({
# #     #                 "emotion": emotion,
# #     #                 "score": score,
# #     #                 "bbox": [x1, y1, x2, y2]
# #     #             })
                
# #     #             # 3. CHECK BUFFER SIZE
# #     #             if len(self.track_emotion_buffer[current_id]) >= VOTE_COUNT:
# #     #                 emotions_list = [e['emotion'] for e in self.track_emotion_buffer[current_id]]
                    
# #     #                 # --- HAPPY PRIORITY LOGIC ---
# #     #                 if 'Happy' in emotions_list:
# #     #                     final_emotion = 'Happy'
# #     #                 else:
# #     #                     final_emotion = Counter(emotions_list).most_common(1)[0][0]
                    
# #     #                 # Find best score for final emotion
# #     #                 best_entry = max(
# #     #                     [e for e in self.track_emotion_buffer[current_id] if e['emotion'] == final_emotion],
# #     #                     key=lambda x: x['score']
# #     #                 )
                    
# #     #                 # 4. PUSH AND LOCK
# #     #                 self.processed_track_ids.add(current_id)
# #     #                 del self.track_emotion_buffer[current_id]
                    
# #     #                 logger.info(f"Final Decision for ID {current_id}: {final_emotion}")
                    
# #     #                 emotion_events.append({
# #     #                     "track_id": current_id,
# #     #                     "emotion": final_emotion,
# #     #                     "score": round(best_entry['score'], 2),
# #     #                     "bbox": best_entry['bbox']
# #     #                 })
        
# #     #     # Clean up old data regardless of detection
# #     #     self.clean_old_ids(current_active_ids)
        
# #     #     if emotion_events:
# #     #         self.send_alert(frame, emotion_events, timestamp)

# #     #     # Visualization Logic
# #     #     if VISUALIZE and hasattr(self, 'vis_queue'):
# #     #         try:
# #     #             vis_frame = self.draw_visualization(frame)
# #     #             _, img_encoded = cv2.imencode('.jpg', vis_frame)
# #     #             frame_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
                
# #     #             buffer_info = []
# #     #             for tid, emotions in self.track_emotion_buffer.items():
# #     #                 buffer_info.append({
# #     #                     "track_id": tid,
# #     #                     "vote_count": len(emotions),
# #     #                     "latest_emotion": emotions[-1]["emotion"] if emotions else "Unknown",
# #     #                 })

# #     #             vis_payload = {
# #     #                 "camera_id": CAMERA_ID,
# #     #                 "timestamp": timestamp,
# #     #                 "frame": frame_b64,
# #     #                 "active_tracks": len(self.track_emotion_buffer),
# #     #                 "processed_count": len(self.processed_track_ids),
# #     #                 "buffer_info": buffer_info,
# #     #             }
# #     #             self.vis_queue.push(json.dumps(vis_payload))
# #     #         except Exception as e:
# #     #             logger.error(f"Visualization error: {e}")
    
# #     def process_frame(self, frame, timestamp):
# #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         frame_h, frame_w = frame.shape[:2]
        
# #         masked_frame = self.mask_frame(frame)  
        
# #         results = self.face_model(masked_frame, verbose=False)
        
# #         emotion_events = []
# #         if results and results[0].boxes is not None and len(results[0].boxes) > 0:
# #             boxes = results[0].boxes.xyxy.numpy()
# #             confs = results[0].boxes.conf.numpy()
            
# #             for box, conf in zip(boxes, confs):
# #                 if conf < Config.FACE_CONFIDENCE_THRESHOLD:
# #                     continue
                
# #                 x1, y1, x2, y2 = map(int, box)
                
# #                 # Apply padding
# #                 pad_x = int((x2 - x1) * Config.PADDING_PERCENT)
# #                 pad_y = int((y2 - y1) * Config.PADDING_PERCENT)
                
# #                 x1_p = max(0, x1 - pad_x)
# #                 y1_p = max(0, y1 - pad_y)
# #                 x2_p = min(frame_w, x2 + pad_x)
# #                 y2_p = min(frame_h, y2 + pad_y)
                
# #                 # Crop face
# #                 face_crop = frame_rgb[y1_p:y2_p, x1_p:x2_p]
# #                 if face_crop.size == 0:
# #                     continue
                
# #                 # Prepare input for Emotion Model
# #                 face_pil = Image.fromarray(face_crop)
# #                 face_tensor = self.transform(face_pil).unsqueeze(0).to(Config.DEVICE)
                
# #                 if self.emotion_model is not None:
# #                     with torch.no_grad():
# #                         outputs = self.emotion_model(face_tensor).logits
# #                         probs = torch.softmax(outputs, dim=1)
# #                         confidence, predicted = torch.max(probs, 1)
                        
# #                         emotion = Config.CLASS_NAMES[predicted.item()]
# #                         score = confidence.item()
# #                 else:
# #                     logger.error("4-bit model not loaded!")
# #                     continue
                
# #                 emotion_events.append({
# #                     "emotion": emotion,
# #                     "score": round(score, 2),
# #                     "bbox": [x1, y1, x2, y2]
# #                 })
        
# #         if emotion_events:
# #             self.send_alert(frame, emotion_events, timestamp)

# #         # Visualization Logic
# #         if VISUALIZE and hasattr(self, 'vis_queue'):
# #             try:
# #                 vis_frame = self.draw_visualization(frame)
# #                 _, img_encoded = cv2.imencode('.jpg', vis_frame)
# #                 frame_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
                
# #                 vis_payload = {
# #                     "camera_id": CAMERA_ID,
# #                     "timestamp": timestamp,
# #                     "frame": frame_b64,
# #                     "detection_count": len(emotion_events),
# #                 }
# #                 self.vis_queue.push(json.dumps(vis_payload))
# #             except Exception as e:
# #                 logger.error(f"Visualization error: {e}")


    
# #     def send_alert(self, frame, events, timestamp):
# #         try:
# #             alert_id = str(uuid.uuid4())
            
# #             # Check if any detected emotion is 'Happy'
# #             happy_events = [e for e in events if e['emotion'] == 'Happy']
            
# #             # 1. Base payload (Always goes to main stats)
# #             main_payload = {
# #                 "camera_id": CAMERA_ID,
# #                 "timestamp": timestamp,
# #                 "emotions": events,
# #                 "description": f"Detected {len(events)} exiting people (Mode).",
# #                 "target_collection": TARGET_COLLECTION,
# #                 "ticket_id": alert_id
# #             }
# #             # Push to main stats immediately
# #             self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(main_payload)})
            
# #             # 2. Alert Payload (Only if Happy)
# #             if happy_events:
# #                 # We create a separate payload for the alert
# #                 alert_payload = main_payload.copy()
                
# #                 # Encode RAW frame (without visualization) first
# #                 _, raw_buffer = cv2.imencode('.jpg', frame)
# #                 raw_frame_b64 = base64.b64encode(raw_buffer).decode('utf-8')
                
# #                 # Now create annotated frame
# #                 annotated_frame = frame.copy()
# #                 for event in events:
# #                     x1, y1, x2, y2 = event['bbox']
# #                     emotion = event['emotion']
# #                     score = event['score']
                    
# #                     # Highlight Happy faces specifically in Green, others in White
# #                     color = (0, 255, 0) if emotion == 'Happy' else (255, 255, 255)
# #                     thickness = 3 if emotion == 'Happy' else 2
                    
# #                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
# #                     ptype = event.get('person_type', 'customer')
# #                     label = f"{ptype} {emotion} ({score:.2f})"
# #                     if emotion == 'Happy':
# #                         label = "ALERTA: HAPPY FACE! " + label
                        
# #                     cv2.putText(annotated_frame, label, (x1, y1 - 10), 
# #                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# #                 # Convert annotated frame to base64 for storage/alert
# #                 _, buffer = cv2.imencode('.jpg', annotated_frame)
# #                 frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
# #                 # Special alert payload properties
# #                 alert_payload["ticket_id"] = alert_id
# #                 alert_payload["frame"] = frame_b64
# #                 alert_payload["frame_without_visualization"] = raw_frame_b64
# #                 alert_payload["description"] = f"Happy face detected! Total exiting: {len(events)}"
                
# #                 # 1. RAISE TICKET
# #                 alert_payload["type"] = "raised"
# #                 alert_payload["target_collection"] = "alert_collection" 
                
# #                 logger.info(f"WOW! Happy face detected. Raising Alert ID: {alert_id}")
# #                 self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(alert_payload)})

# #                 # 2. INSTANTLY RESOLVE TICKET
# #                 # We reuse the same payload but update type and description for resolution
# #                 resolve_payload = alert_payload.copy()
# #                 resolve_payload["type"] = "resolved"
# #                 resolve_payload["description"] = f"Happy face verified. Auto-resolved: {alert_id}"
                
# #                 logger.info(f"Auto-resolving Alert ID: {alert_id}")
# #                 self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(resolve_payload)})

# #             logger.info(f"Event Pushed. Total people: {len(events)}. Happy: {len(happy_events) > 0}")
            
# #         except Exception as e:
# #             logger.error(f"Failed to push alert: {e}")

# #     def run(self):
# #         logger.info(f"Starting Service. Target Camera: {CAMERA_SOURCE}")
# #         src = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
        
# #         while True:
# #             try:
# #                 cap = cv2.VideoCapture(src)
# #                 if not cap.isOpened():
# #                     logger.warning(f"Unable to open camera: {src}. Retrying...")
# #                     time.sleep(5)
# #                     continue
                
# #                 logger.info(f"Camera connected: {src}")
# #                 source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
# #                 frame_interval = max(1, int(source_fps / TARGET_FPS))
# #                 frame_count = 0

# #                 while True:
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         break

# #                     if frame_count % frame_interval == 0:
# #                         try:
# #                             timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
# #                             self.process_frame(frame, timestamp)
                            
# #                             # Push visualization frame if enabled
# #                             if VISUALIZE and hasattr(self, 'vis_queue'):
# #                                 try:
# #                                     # Draw visualization on frame
# #                                     vis_frame = self.draw_visualization(frame)
                                    
# #                                     _, img_encoded = cv2.imencode('.jpg', vis_frame)
# #                                     frame_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
                                    
# #                                     # Get current buffer states for display
# #                                     buffer_info = []
# #                                     for tid, emotions in self.track_emotion_buffer.items():
# #                                         buffer_info.append({
# #                                             "track_id": tid,
# #                                             "vote_count": len(emotions),
# #                                             "latest_emotion": emotions[-1]["emotion"] if emotions else "Unknown",
# #                                         })
                                    
# #                                     vis_payload = {
# #                                         "camera_id": CAMERA_ID,
# #                                         "timestamp": timestamp,
# #                                         "frame": frame_b64,
# #                                         "frame_num": frame_count,
# #                                         "active_tracks": len(self.track_emotion_buffer),
# #                                         "processed_count": len(self.processed_track_ids),
# #                                         "buffer_info": buffer_info,
# #                                     }
# #                                     self.vis_queue.push(json.dumps(vis_payload))
# #                                 except Exception as e:
# #                                     logger.error(f"Failed to push visualization frame: {e}")
                                    
# #                         except Exception as e:
# #                             logger.error(f"Processing error: {e}")

# #                     frame_count += 1
                
# #                 cap.release()
                
# #             except Exception as e:
# #                 logger.error(f"Video loop error: {e}. Retrying...")
# #                 if 'cap' in locals() and cap.isOpened():
# #                     cap.release()
# #                 time.sleep(5)

# # if __name__ == "__main__":
# #     service = DirectEmotionService()
# #     service.run()


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
from collections import deque, Counter
import uuid
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import ViTForImageClassification, BitsAndBytesConfig
import bitsandbytes

# AI Models
from ultralytics import YOLO
load_dotenv()

# --- CONFIGURATION ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

# Stream Config
INPUT_QUEUE = os.getenv("INPUT_QUEUE_NAME", "stream:cam:6")
GROUP_NAME = os.getenv("GROUP_NAME", "cg:cam:exit_emotion_detection_service")
TARGET_FPS = int(os.getenv("TARGET_FPS", "1"))

# Model paths - use local files from service directory
SERVICE_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = Path("/workspace")

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

YOLO_MODEL = os.getenv("YOLO_MODEL", "face_detection_06022026.pt")
BNB_4BIT_PATH = os.getenv("BNB_4BIT_PATH", "bnb_4bit_model")

# Output
OUTPUT_QUEUE_1 = os.getenv("OUTPUT_QUEUE_NAME_1", "emotion_alerts")
TARGET_COLLECTION = os.getenv("TARGET_COLLECTION", "emotion_events")
CAMERA_ID = os.getenv("CAMERA_ID", INPUT_QUEUE.split(":")[-1])
VISUALIZE = str(os.getenv("VISUALIZE", "False")).lower() == "true"

# Constants
EMOTIONS_LABELS = ['neutral', 'happy']
VOTE_COUNT = 3  # Number of frames to collect before deciding the emotion
ACK_TTL_SECONDS = 10 * 60

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


class FixedSizeQueue:
    def __init__(self, redis_client, name, max_size):
        self.r = redis_client
        self.name = name
        self.max_size = max_size

    def push(self, item):
        """Push item to queue. Oldest items are dropped."""
        pipe = self.r.pipeline()
        pipe.rpush(self.name, item)
        pipe.ltrim(self.name, -self.max_size, -1)
        pipe.execute()

    def pop(self, timeout=0):
        """Blocking FIFO pop."""
        result = self.r.blpop(self.name, timeout=timeout)
        if result:
            _, value = result
            return value
        return None

    def size(self):
        return self.r.llen(self.name)


class Config:
    FRAME_INTERVAL = 1  # Process 1 frame per second
    FACE_CONFIDENCE_THRESHOLD = 0.4
    PADDING_PERCENT = 0.3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BNB_4BIT_PATH = os.getenv("BNB_4BIT_PATH", "bnb_4bit_model")
    ID2LABEL = {0: 'neutral', 1: 'happy'}
    LABEL2ID = {'neutral': 0, 'happy': 1}
    CLASS_NAMES = ['Neutral', 'Happy']


class DirectEmotionService:
    def __init__(self):
        self.settings = self.load_settings()
        self.redis = self.connect_redis()
        self.results = []
        self.processed_track_ids = set()
        self.track_emotion_buffer = {}
        self.detection_history = []  # Track recently alerted detections spatially to prevent repeats

        logger.info("Loading YOLO model")
        self.face_model = YOLO(YOLO_MODEL)

        logger.info("Loading Emotion model")
        self.emotion_model = self.load_models()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if VISUALIZE:
            self.vis_queue = FixedSizeQueue(
                self.redis, f"stream:visualization:exit_emotion:{CAMERA_ID}", 10
            )
            logger.info(f"Visualization queue initialized: stream:visualization:exit_emotion:{CAMERA_ID}")

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

    def mask_frame(self, frame):
        height, width = frame.shape[:2]
        polygons = self.settings.get("1", {}).get("polygons", [])
        mask = np.zeros((height, width), dtype=np.uint8)

        for poly in polygons:
            points = []
            for p in poly.get("normalized", []):
                x = p["x"]
                y = p["y"]
                # Handle both normalized (0-1) and absolute coordinates
                px = int(x) if x > 1.0 else int(x * width)
                py = int(y) if y > 1.0 else int(y * height)
                points.append([px, py])
            
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result

    def clean_old_ids(self, current_active_ids):
        """Clean up buffers for people who left the screen."""
        buffer_ids = list(self.track_emotion_buffer.keys())
        for pid in buffer_ids:
            if pid not in current_active_ids:
                del self.track_emotion_buffer[pid]

    def load_models(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        try:
            bnb_4bit_model = ViTForImageClassification.from_pretrained(
                Config.BNB_4BIT_PATH,
                quantization_config=quantization_config,
                device_map="auto"
            )
            return bnb_4bit_model
        except Exception as e:
            logger.error(f"Failed to load user model: {e}")
            return None

    def download_image(self, url):
        """Download image from URL (served by MinIO via stream_handling_service)."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None

    def mark_acked(self, stream: str, msg_id: str, consumer: str):
        """Mark a message as acknowledged by this consumer."""
        cnt_key = f"ack:cnt:{stream}:{msg_id}"
        seen_key = f"ack:seen:{stream}:{msg_id}"
        pipe = self.redis.pipeline()

        pipe.sadd(seen_key, consumer)
        pipe.expire(seen_key, ACK_TTL_SECONDS)
        results = pipe.execute()
        was_added = results[0]

        if was_added:
            pipe = self.redis.pipeline()
            pipe.incr(cnt_key)
            pipe.expire(cnt_key, ACK_TTL_SECONDS)
            pipe.execute()

    def process_message(self, message: dict):
        """Process a single Redis stream message containing a frame URL."""
        message = {k.decode(): v.decode() for k, v in message.items()}
        frame_url = message.get("frame_url")
        camera_id = message.get("camera_id", CAMERA_ID)
        timestamp = message.get("timestamp", time.strftime('%Y-%m-%d %H:%M:%S'))

        if not frame_url:
            logger.warning("Message missing 'frame_url'. Skipping.")
            return

        frame = self.download_image(frame_url)
        if frame is None:
            logger.error(f"Could not download frame from {frame_url}")
            return

        self.process_frame(frame, timestamp)

    def process_frame(self, frame, timestamp):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]

        masked_frame = self.mask_frame(frame)
        results = self.face_model(masked_frame, verbose=False)

        emotion_events = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                if conf < Config.FACE_CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box)

                pad_x = int((x2 - x1) * Config.PADDING_PERCENT)
                pad_y = int((y2 - y1) * Config.PADDING_PERCENT)

                x1_p = max(0, x1 - pad_x)
                y1_p = max(0, y1 - pad_y)
                x2_p = min(frame_w, x2 + pad_x)
                y2_p = min(frame_h, y2 + pad_y)

                face_crop = frame_rgb[y1_p:y2_p, x1_p:x2_p]
                if face_crop.size == 0:
                    continue

                face_pil = Image.fromarray(face_crop)
                face_tensor = self.transform(face_pil).unsqueeze(0).to(Config.DEVICE)

                if self.emotion_model is not None:
                    with torch.no_grad():
                        outputs = self.emotion_model(face_tensor).logits
                        probs = torch.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probs, 1)

                        emotion = Config.CLASS_NAMES[predicted.item()]
                        score = confidence.item()
                else:
                    logger.error("4-bit model not loaded!")
                    continue

                emotion_events.append({
                    "emotion": emotion,
                    "score": round(score, 2),
                    "bbox": [x1, y1, x2, y2]
                })

        if emotion_events:
            self.send_alert(frame, emotion_events, timestamp)

        if VISUALIZE and hasattr(self, 'vis_queue'):
            try:
                vis_frame = self.draw_visualization(frame, emotion_events)
                _, img_encoded = cv2.imencode('.jpg', vis_frame)
                frame_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

                vis_payload = {
                    "camera_id": CAMERA_ID,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "detection_count": len(emotion_events),
                }
                self.vis_queue.push(json.dumps(vis_payload))
            except Exception as e:
                logger.error(f"Visualization error: {e}")

    def draw_visualization(self, frame, events):
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        # 1. Draw Polygons (Mask)
        # Using "1" as key to match mask_frame implementation
        polygons = self.settings.get("1", {}).get("polygons", [])
        
        mask_overlay = vis_frame.copy()
        for poly in polygons:
            points = []
            for p in poly.get("normalized", []):
                x = p["x"]
                y = p["y"]
                # Handle both normalized (0-1) and absolute coordinates
                px = int(x) if x > 1.0 else int(x * width)
                py = int(y) if y > 1.0 else int(y * height)
                points.append([px, py])
                
            points = np.array(points, dtype=np.int32)
            
            # Draw polygon outline in Blue
            cv2.polylines(vis_frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)
            
            # Highlight the active area (inside polygon) slightly? 
            # Or darken the outside? mask_frame blackens the outside.
            # Let's just draw the polygon for now as requested.
            
        # 2. Draw Detections
        for event in events:
            x1, y1, x2, y2 = event['bbox']
            emotion = event['emotion']
            score = event.get('score', 0.0)
            
            # Color coding: Happy=Green, others=Orange
            color = (0, 255, 0) if emotion == 'Happy' else (0, 165, 255)
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{emotion} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Label background
            cv2.rectangle(vis_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                       
        return vis_frame

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        width_inter = max(0, x_inter2 - x_inter1)
        height_inter = max(0, y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        area_box1 = (x2 - x1) * (y2 - y1)
        area_box2 = (x4 - x3) * (y4 - y3)
        area_union = area_box1 + area_box2 - area_inter
        return area_inter / area_union if area_union > 0 else 0

    def send_alert(self, frame, events, timestamp):
        try:
            alert_id = str(uuid.uuid4())
            
            # --- SPATIAL SUPPRESSION LOGIC (ALL EVENTS) ---
            now = time.time()
            # Clean history older than 30 seconds (reduces duration but prevents overlap)
            self.detection_history = [h for h in self.detection_history if now - h['time'] < 30]
            
            new_events = []
            for event in events:
                is_duplicate = False
                for past in self.detection_history:
                    # If IOU > 0.5, we consider it the same person/region
                    if self.calculate_iou(event['bbox'], past['bbox']) > 0.5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    new_events.append(event)
            # ----------------------------------------------

            # If no new detections, skip everything to avoid repeating alerts/logs
            if not new_events:
                return

            # Update history with new detections
            for ne in new_events:
                self.detection_history.append({'bbox': ne['bbox'], 'time': now})

            happy_events = [e for e in new_events if e['emotion'] == 'Happy']

            main_payload = {
                "camera_id": CAMERA_ID,
                "timestamp": timestamp,
                "emotions": new_events, # Only send unique new detections
                "description": f"Detected {len(new_events)} exiting people.",
                "target_collection": TARGET_COLLECTION,
                "ticket_id": alert_id
            }
            
            # Push to main stats
            self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(main_payload)})

            # Raise formal ticket if any of the NEW detections are Happy
            if happy_events:
                alert_payload = main_payload.copy()
                _, raw_buffer = cv2.imencode('.jpg', frame)
                raw_frame_b64 = base64.b64encode(raw_buffer).decode('utf-8')

                annotated_frame = frame.copy()
                for event in events: # Draw all current for context, but alert is for new happy
                    x1, y1, x2, y2 = event['bbox']
                    emotion = event['emotion']
                    score = event['score']
                    color = (0, 255, 0) if emotion == 'Happy' else (255, 255, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{emotion} ({score:.2f})"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                alert_payload["ticket_id"] = alert_id
                alert_payload["frame"] = frame_b64
                alert_payload["frame_without_visualization"] = raw_frame_b64
                alert_payload["description"] = f"Happy face detected! Total: {len(new_events)}"
                alert_payload["type"] = "raised"
                alert_payload["target_collection"] = "alert_collection"

                logger.info(f"Raising Alert: Happy face detected. ID: {alert_id}")
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(alert_payload)})

                resolve_payload = alert_payload.copy()
                resolve_payload["type"] = "resolved"
                resolve_payload["description"] = f"Auto-resolved: {alert_id}"
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(resolve_payload)})

            logger.info(f"Event Pushed. Total people: {len(new_events)}. Happy: {len(happy_events) > 0}")

        except Exception as e:
            logger.error(f"Failed to push alert: {e}")

    def run(self):
        logger.info(f"Starting Service. Consuming from Redis stream: {INPUT_QUEUE}, group: {GROUP_NAME}")
        while True:
            try:
                resp = self.redis.xreadgroup(
                    groupname=GROUP_NAME,
                    consumername="worker-1",
                    streams={INPUT_QUEUE: ">"},
                    count=1,
                    block=5000,
                )

                if resp:
                    stream_name, messages = resp[0]
                    msg_id, message = messages[0]
                    self.process_message(message)
                    self.redis.xack(INPUT_QUEUE, GROUP_NAME, msg_id)
                    self.mark_acked(INPUT_QUEUE, msg_id.decode(), GROUP_NAME)

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message from input queue.")
            except Exception as e:
                logger.exception(f"Unexpected error in main loop: {e}")
                time.sleep(1)


if __name__ == "__main__":
    service = DirectEmotionService()
    service.run()

