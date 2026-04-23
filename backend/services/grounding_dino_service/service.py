import io
import torch
import requests
from PIL import Image
from litserve import LitAPI, LitServer
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, HqqConfig
import cv2
import numpy as np
import os
from dotenv import load_dotenv
import logging
import logging_loki
from multiprocessing import Queue

load_dotenv()

# Configure Logging
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

logger = logging.getLogger("grounding_dino_service")
logger.setLevel(logging.INFO)

# Always add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)

# Try to add Loki handler, but don't fail if it's unavailable
try:
    loki_handler = logging_loki.LokiQueueHandler(
        Queue(-1),
        url=LOKI_URL,
        tags={"application": "grounding_dino_service"},
        version="1",
    )
    logger.addHandler(loki_handler)
    logger.info("Loki logging handler initialized successfully")
except Exception as e:
    logger.warning(
        f"Failed to initialize Loki handler: {e}. Continuing with console logging only."
    )


is_cuda_available = torch.cuda.is_available()
# Get device from env var, or auto-detect (cuda if available, else cpu)
device_str = os.getenv("DEVICE", "cuda" if is_cuda_available else "cpu")
DEVICE = torch.device(device_str)
logger.info(f"Using device: {DEVICE}")

ROTATION_DICT = {
    "ROTATE_90_CLOCKWISE": cv2.ROTATE_90_CLOCKWISE,
    "ROTATE_90_COUNTERCLOCKWISE": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "ROTATE_180": cv2.ROTATE_180,
}

quant_config = HqqConfig(nbits=4, group_size=64)


class GroundingDinoService(LitAPI):
    def setup(self, device=DEVICE):
        try:
            logger.info("Starting Grounding DINO service setup")
            self.device = device
            self.model_path = "/models/grounding_dino"
            self.hf_model_id = "IDEA-Research/grounding-dino-base"

            # Check if model exists locally
            if not os.path.exists(os.path.join(self.model_path, "config.json")):
                logger.info(f"Model not found at {self.model_path}. Downloading from {self.hf_model_id}...")
                
                # Ensure directory exists
                os.makedirs(self.model_path, exist_ok=True)
                
                # Download and save Processor
                processor = AutoProcessor.from_pretrained(self.hf_model_id)
                processor.save_pretrained(self.model_path)
                
                # Download and save Model
                model = AutoModelForZeroShotObjectDetection.from_pretrained(self.hf_model_id)
                model.save_pretrained(self.model_path)
                logger.info("Model downloaded and saved locally.")
            else:
                logger.info(f"Found local model at {self.model_path}")

            logger.info(f"Loading processor from {self.model_path}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True
            )

            logger.info(f"Loading model from {self.model_path} with 4-bit quantization")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                local_files_only=True
            ).to(self.device)
            self.model.eval()

            logger.info(
                f"Grounding DINO service setup complete on device: {self.device}"
            )
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}", exc_info=True)
            raise

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
        try:
            text = request.get("text", "")

            if "image_url" in request:
                image_url = request["image_url"]
                polygon_mask = request.get("polygon_mask", None)
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
                frame = np.array(image)
                frame = cv2.rotate(frame, ROTATION_DICT[request.get("rotation")])
                image = Image.fromarray(frame)
            if polygon_mask is not None:
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # apply all polygons at once
                masked = self.apply_polygon_masks(frame, polygon_mask)
                masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(masked)
                
            return image, text

        except Exception as e:
            logger.error(f"Error in decode_request: {str(e)}", exc_info=True)
            raise

    def predict(self, inputs):
        try:
            image, text = inputs
            logger.info(f"Running prediction with text prompt: {text}")

            inputs = self.processor(images=image, text=text, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = self.model(**inputs)

            target_sizes = [image.size[::-1]]
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, text_threshold=0.3, target_sizes=target_sizes
            )[0]

            results["scores"] = results["scores"].cpu().detach().tolist()
            results["boxes"] = results["boxes"].cpu().detach().tolist()

            logger.info(
                f"Prediction successful. Found {len(results['scores'])} detections"
            )
            return results

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise

    def encode_response(self, prediction):
        try:
            logger.info("Encoding response")
            return {"detections": prediction}
        except Exception as e:
            logger.error(f"Error encoding response: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    server = LitServer(GroundingDinoService(), accelerator="auto")
    port = int(os.getenv("PORT", "8081"))
    logger.info(f"Starting server on port {port}")
    server.run(port=port)
