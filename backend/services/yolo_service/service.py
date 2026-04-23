from ultralytics import YOLO
from litserve import LitServer, LitAPI
import torch
from PIL import Image
import requests
import io
import logging
import logging_loki
from multiprocessing import Queue
import cv2
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "yolo_service"},
    version="1",
)

logger = logging.getLogger("yolo_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Also log to console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

is_cuda_available = torch.cuda.is_available()
device_str = os.getenv("DEVICE", "cuda" if is_cuda_available else "cpu")
DEVICE = torch.device(device_str)
logger.info(f"Using device: {DEVICE}")
# Use local model file if it exists, otherwise fall back to MODEL_ID
LOCAL_MODEL_PATH = "/workspace/plate_detection_model_19_01_2026.pt"
MODEL_ID = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else os.getenv("MODEL_ID")
logger.info(f"using model: {MODEL_ID}")


ROTATION_DICT={
    "ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,
    "ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,
    "ROTATE_180":cv2.ROTATE_180
}

class YoloObjectDetector(LitAPI):
    def setup(self, device=DEVICE):
        self.device = device
        self.model_id = MODEL_ID
        self.model = YOLO(self.model_id).to(self.device)
        logger.info(f"Model {self.model_id} loaded on {self.device}")

    def apply_polygon_masks(self, frame, polygons):
        """
        frame: numpy array (H,W,3)
        polygons: list of polygons, each polygon is list of [x,y] points
        """
        # Convert each polygon to np.int32 array
        polygons = [np.array(p, dtype=np.int32) for p in polygons]

        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Fill all polygons at once
        cv2.fillPoly(mask, polygons, 255)

        # Invert mask → region to remove becomes 255
        mask_inv = cv2.bitwise_not(mask)

        # Apply mask (remove polygon areas)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
        return masked_frame

    def decode_request(self, request):
        # Extract polygon_mask before branches to avoid UnboundLocalError
        polygon_mask = request.get("polygon_mask", None)
        
        if "image_url" in request:
            image_url = request["image_url"]
            logger.info(f"Received request with image URL: {image_url}")
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                logger.info("Successfully downloaded and processed image from URL")
            except Exception as e:
                logger.error(f"Failed to fetch image from URL: {e}")
                raise e
        else:
            logger.info("Received request with image file upload")
            image_file = request["image"]
            image_bytes = image_file.file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.info("Successfully processed uploaded image file")
        if "rotation" in request:
            frame=np.array(image)
            frame=cv2.rotate(frame,ROTATION_DICT[request.get("rotation")])
            image=Image.fromarray(frame)
        if polygon_mask is not None:
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # apply all polygons at once
            masked = self.apply_polygon_masks(frame, polygon_mask)
            masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(masked)
        
        conf = request.get("confidence", 0.25)
        return {"image": image, "conf": conf}

    def predict(self, data):
        logger.info("Starting prediction")
        image = data["image"]
        conf = data["conf"]
        with torch.no_grad():
            outputs = self.model(image, conf=conf)
        outputs = [out.to_json() for out in outputs]
        logger.info(f"Prediction complete. Detected {len(outputs)} frames/outputs")
        return outputs

    def encode_response(self, output):
        logger.info("Encoding response")
        return {"predict": output}


if __name__ == "__main__":
    server = LitServer(YoloObjectDetector(), api_path="/predict", accelerator="auto")
    server.run(port=os.getenv("PORT"))