import os
# Force Rebuild 123
import gc
import io
import logging
from typing import List, Dict

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFile
import requests
import numpy as np
from litserve import LitServer, LitAPI
from dotenv import load_dotenv
import logging_loki
from multiprocessing import Queue
from fastai.vision.all import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

load_dotenv()

# Configuration
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
MODEL_PATH = os.getenv("MODEL_PATH", "classifier_customer_vs_staff.pkl")
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(1000),
    url=LOKI_URL,
    tags={"application": "staff_customer_classification_service"},
    version="1",
)

logger = logging.getLogger("staff_customer_classification_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

class StaffCustomerClassifier(LitAPI):
    def setup(self, device=DEVICE):
        self.device = device
        self.model_path = MODEL_PATH
        logger.info(f"Setting up model on {self.device}. Model path: {self.model_path}")
        
        try:
            self.learn = load_learner(self.model_path)
            self.learn.model.to(self.device)
            self.learn.model.eval()
            self.class_names = list(self.learn.dls.vocab)
            logger.info(f"Model loaded successfully with classes: {self.class_names}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            # Fallback for build/init if model is missing
            self.class_names = ["customer", "staff"] # Dummy classes
            self.learn = None

    def decode_request(self, request):
        logger.info("NEW CODE RUNNING: decode_request called")
        images = []
        
        bboxes = request.get("bboxes")
        if bboxes and isinstance(bboxes, list):
            # Batch processing
            image_url = request.get("image_url")
            if not image_url:
                raise ValueError("image_url required for batch processing")
            
            logger.info(f"Processing batch of {len(bboxes)} bboxes from URL: {image_url}")
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                base_image = Image.open(io.BytesIO(response.content)).convert("RGB")
                width, height = base_image.size
            except Exception as e:
                logger.error(f"Failed to fetch image: {e}")
                raise e
            
            # Crop each bbox
            for bbox in bboxes:
                try:
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(width, int(x2))
                    y2 = min(height, int(y2))
                    
                    if x2 > x1 and y2 > y1:
                        cropped = base_image.crop((x1, y1, x2, y2))
                        images.append(cropped)
                    else:
                        logger.warning(f"Invalid bbox dimensions: {bbox}, using full image")
                        images.append(base_image.copy())
                except Exception as e:
                    logger.error(f"Failed to crop bbox {bbox}: {e}")
                    images.append(base_image.copy())
        
        else:
            # Single request
            image = None
            if "image_url" in request:
                image_url = request["image_url"]
                logger.info(f"Processing image URL: {image_url}")
                try:
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    logger.error(f"Failed to fetch image: {e}")
                    raise e
            elif "image" in request:
                logger.info("Processing uploaded image file")
                image_file = request["image"]
                image_bytes = image_file.file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            if image is None:
                raise ValueError("No image provided (use 'image_url' or 'image' file)")

            # Handle cropping if bbox provided
            bbox = request.get("bbox")
            if bbox:
                try:
                    if isinstance(bbox, str):
                        import json
                        bbox = json.loads(bbox)
                    
                    x1, y1, x2, y2 = bbox
                    width, height = image.size
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(width, int(x2))
                    y2 = min(height, int(y2))
                    
                    if x2 > x1 and y2 > y1:
                        logger.info(f"Cropping image to bbox: {x1, y1, x2, y2}")
                        image = image.crop((x1, y1, x2, y2))
                    else:
                        logger.warning(f"Invalid bbox dimensions: {bbox}, skipping crop.")
                except Exception as e:
                    logger.error(f"Failed to apply bbox crop: {e}")
            
            images.append(image)

        # Prepare transforms
        t = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

        # Convert images to Tensor stack
        if not images:
             return torch.empty(0, 3, IMG_SIZE, IMG_SIZE)
        
        tensor_list = []
        for img in images:
            try:
                tensor_list.append(t(img))
            except Exception as e:
                logger.error(f"Failed to transform image: {e}")
        
        if not tensor_list:
             return torch.empty(0, 3, IMG_SIZE, IMG_SIZE)

        return torch.stack(tensor_list)

    def predict(self, inputs):
        # inputs is a list of outputs from decode_request.
        # Since decode_request returns a list of images (for one request),
        # inputs is a list of list of images: [[img1_a, img1_b], [img2_a], ...]
        
        batch_results = []
        
        positive_class_idx = -1
        if "staff" in self.class_names:
            positive_class_idx = self.class_names.index("staff")

        for request_tensor in inputs:
            request_results = []
            
            if self.learn is None:
                # If we don't know N, just return error per tensor?
                # But we can't easily match N without shape
                n_crops = request_tensor.shape[0] if isinstance(request_tensor, torch.Tensor) else 0
                batch_results.append([{"error": "Model not loaded"}] * n_crops)
                continue
            
            # Litserve might unpack the Tensor into a list of Tensors (rows)
            crops_list = []
            if isinstance(request_tensor, list):
                crops_list = request_tensor
            elif isinstance(request_tensor, torch.Tensor):
                # If it wasn't unpacked (maybe batch size 1 or config dependent)
                crops_list = [request_tensor[i] for i in range(request_tensor.shape[0])]
            else:
                 logger.error(f"Expected Tensor or List[Tensor] but got {type(request_tensor)}")
                 batch_results.append([{"error": "Invalid input type"}])
                 continue

            # Iterate over crops
            to_pil = transforms.ToPILImage()
            for i, img_tensor in enumerate(crops_list):
                try:
                    # Convert to numpy uint8 array (H, W, C)
                    # This mimics a standard image loaded by PIL/cv2
                    img_np = (img_tensor.permute(1, 2, 0) * 255).byte().numpy()

                    # Use fastai's predict
                    pred_class, pred_idx, probs = self.learn.predict(img_np)
                    
                    label = str(pred_class)
                    conf = float(probs[pred_idx])
                    
                    result = {
                        "predicted_label": label,
                        "confidence": conf,
                        "probabilities": {
                            self.class_names[j]: float(probs[j])
                            for j in range(len(self.class_names))
                        }
                    }
                    if positive_class_idx != -1:
                        result["positive_class_score"] = float(probs[positive_class_idx])
                        
                    request_results.append(result)
                    logger.info(f"Prediction: Label='{label}', Confidence={conf:.4f}")
                except Exception as e:
                    logger.error(f"Prediction failed for image {i}: {e}")
                    request_results.append({"error": str(e)})

            batch_results.append(request_results)
        
        logger.info(f"Batch prediction complete. Requests: {len(inputs)}")
        return batch_results

    def encode_response(self, output):
        # Force garbage collection after processing request to free up memory from full frame decoding
        gc.collect() 
        return output

if __name__ == "__main__":
    server = LitServer(StaffCustomerClassifier(), api_path="/predict", accelerator="auto", max_batch_size=4, workers_per_device=1)
    server.run(port=os.getenv("PORT", "8000"))
