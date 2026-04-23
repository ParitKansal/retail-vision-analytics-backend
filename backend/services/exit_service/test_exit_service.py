import cv2
import numpy as np
import logging
import time
import os
from service import DirectEmotionService

# --- CONFIGURATION SECTION ---

# Set this to True for Live Stream, False for Video File
USE_LIVE_STREAM = False 

# Option A: Live Stream URL (RTSP, HTTP, or '0' for webcam)
LIVE_STREAM_URL = "rtsp://192.168.2.127:8554/kiosk"
# LIVE_STREAM_URL = "0"  # Use "0" for local webcam

# Option B: Video File Path
VIDEO_FILE_PATH = None 
# "/Users/xelpmoc/Documents/Xelp_work/McD/Sample_videos/Kiosk_2025-12-05_1.mp4"
# -----------------------------

# --- IMPORT FROM service.py ---
try:
    from service import DirectEmotionService
except ImportError:
    print("CRITICAL ERROR: Could not import 'DirectEmotionService'.")
    print("Please ensure your main script is saved as 'service.py' in the same folder.")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("test_service")

class VisualTestService(DirectEmotionService):
    """
    Inherits from DirectEmotionService to override Redis
    and add visual debugging windows.
    """

    def __init__(self):
        super().__init__()

    def connect_redis(self):
        # Disable Redis for testing
        logger.info("TEST MODE: Redis connection disabled.")
        return None

    def send_alert(self, frame, events, timestamp):
        # Print alerts to console instead of Redis
        print(f"\n[ALERT TRIGGERED] {timestamp}")
        for event in events:
            print(f" > Track ID: {event['track_id']} | Emotion: {event['emotion']} ({event['score']})")

    def draw_zone(self, frame):
        h, w = frame.shape[:2]
        polygons = self.settings.get("1", {}).get("polygons", [])
        
        for poly in polygons:
            points = np.array([
                [int(p["x"] * w), int(p["y"] * h)] 
                for p in poly.get("normalized", [])
            ], dtype=np.int32)
            
            cv2.polylines(frame, [points], isClosed=True, color=(255, 255, 0), thickness=2)
            if len(points) > 0:
                cv2.putText(frame, poly["name"], tuple(points[0]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def process_and_visualize(self, frame):
        h, w = frame.shape[:2]
        
        # 1. Draw Zone
        self.draw_zone(frame)

        # 2. Run YOLO Tracking
        results = self.tracking_model.track(frame, persist=True, verbose=False, classes=0)
        
        if not results or results[0].boxes.id is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Check if feet are in zone
            feet_point = (int((x1 + x2) / 2), int(y2))
            in_zone = self.check_zone(feet_point, frame.shape)
            
            # Colors: Green (Tracking), Red (In Zone/Processing)
            color = (0, 0, 255) if in_zone else (0, 255, 0)
            
            # Draw Person Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3. Emotion Logic (Only if inside zone)
            if in_zone:
                safe_y1, safe_y2 = max(0, y1), min(h, y2)
                safe_x1, safe_x2 = max(0, x1), min(w, x2)
                
                body_roi = frame[safe_y1:safe_y2, safe_x1:safe_x2]
                
                if body_roi.size > 0:
                    rgb_body = cv2.cvtColor(body_roi, cv2.COLOR_BGR2RGB)
                    
                    # Detect Faces
                    faces = self.mtcnn_detector.detect_faces(rgb_body)
                    
                    if faces:
                        best_face = max(faces, key=lambda x: x['confidence'])
                        fx, fy, fw, fh = best_face['box']
                        
                        # Face crop
                        face_img = body_roi[max(0, fy):fy+fh, max(0, fx):fx+fw]
                        
                        # Predict
                        emotion, score = self.predict_emotion(face_img)

                        # Draw Visuals
                        global_fx = safe_x1 + fx
                        global_fy = safe_y1 + fy
                        
                        cv2.rectangle(frame, (global_fx, global_fy), 
                                      (global_fx + fw, global_fy + fh), (255, 0, 0), 2)
                        
                        label = f"{emotion} {int(score*100)}%"
                        cv2.putText(frame, label, (global_fx, global_fy - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        # Show Debug Face Window
                        if face_img.size > 0:
                            try:
                                cv2.imshow(f"Face_Debug", cv2.resize(face_img, (150, 150)))
                            except: pass

        return frame

    def run_test(self):
        # Determine Source
        if USE_LIVE_STREAM:
            source = int(LIVE_STREAM_URL) if LIVE_STREAM_URL.isdigit() else LIVE_STREAM_URL
            source_name = "Live Stream"
        else:
            source = VIDEO_FILE_PATH
            source_name = "Video File"
            if not os.path.exists(source):
                logger.error(f"Video file not found: {source}")
                return

        logger.info(f"Starting Visual Test on: {source_name} ({source})")
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open source: {source}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Stream ended or failed to read frame.")
                break

            # Resize huge streams for easier viewing
            # frame = cv2.resize(frame, (1280, 720))

            vis_frame = self.process_and_visualize(frame)
            
            cv2.imshow("Test Service (Press 'q' to quit)", vis_frame)

            # '1' ms delay for live streams (fastest processing)
            # '25' ms delay for video files (approx 40fps playback)
            delay = 1 if USE_LIVE_STREAM else 25
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = VisualTestService()
    tester.run_test()