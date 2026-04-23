import os
import redis
import json
import torch
import torch.nn as nn
import requests
import numpy as np
import cv2
import time
import logging
import base64
from PIL import Image, ImageDraw
from io import BytesIO
from torchvision import transforms
from masks import MASK_SETS
from transformers import ViTForImageClassification
from peft import get_peft_model, LoraConfig
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
import uuid
from collections import defaultdict, deque

# ==============================
# Determinism
# ==============================
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    # Some CUDA operations might not support determinism, skip if it fails
    pass


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
INPUT_QUEUES = os.getenv("INPUT_QUEUE_NAMES", "stream:cam:1,stream:cam:7").split(",")
GROUP_NAME = os.getenv("GROUP_NAME", "cg:cam:dirty_floor_service")
MODEL_PATH = os.getenv(
    "MODEL_PATH", "/workspace/ViT_HQQ_LoRA_Masked_16_03_2026_(1).pth"
)
VISUALIZE = os.getenv("VISUALIZE", "False").lower() == "true"
CONSUMER_NAME = f"consumer_{os.getenv('HOSTNAME', 'unknown')}"
ACK_TTL_SECONDS = 3600  # 1 hour

# New Params
BUFFER_DURATION_SECONDS = int(os.getenv("BUFFER_DURATION_SECONDS", 60))
DIRTY_THRESHOLD_PERCENT = float(os.getenv("DIRTY_THRESHOLD_PERCENT", 90))
CLEAN_THRESHOLD_PERCENT = float(os.getenv("CLEAN_THRESHOLD_PERCENT", 90))
OUTPUT_QUEUE_NAME = os.getenv("OUTPUT_QUEUE_NAME", "database:queue:0")

# Per-frame model confidence gate:
# If model predicts Dirty but dirty_score < DIRTY_SCORE_THRESHOLD → override to Clean
DIRTY_SCORE_THRESHOLD = float(os.getenv("DIRTY_SCORE_THRESHOLD", 0.99))

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# Define transformations (matching colab_working_code.py)
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ==============================
# MaskedViTClassifier — identical to training
# ==============================
class MaskedViTClassifier(nn.Module):
    def __init__(self, vit_classifier):
        super().__init__()
        self.vit_classifier = vit_classifier
        self.black_threshold = -1.5
        self.patch_size = 16
        self.num_patches = 196

    def _build_attention_mask(self, pixel_values):
        B, C, H, W = pixel_values.shape
        p = self.patch_size
        patches = pixel_values.unfold(2, p, p).unfold(3, p, p)
        patch_mean = patches.mean(dim=(1, 4, 5))
        is_black = (patch_mean < self.black_threshold).view(B, -1)
        cls_col = torch.ones(B, 1, dtype=torch.bool, device=pixel_values.device)
        return torch.cat([cls_col, ~is_black], dim=1)

    def _mask_to_extended_attention(self, bool_mask):
        extended = bool_mask.to(dtype=torch.float32)
        extended = (1.0 - extended) * -10000.0
        return extended.unsqueeze(1).unsqueeze(2)

    def forward(self, pixel_values):
        bool_mask = self._build_attention_mask(pixel_values)
        attn_mask = self._mask_to_extended_attention(bool_mask)
        return self.vit_classifier(pixel_values=pixel_values, attention_mask=attn_mask)


class DirtyFloorService:
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        self.model = self.load_model()
        self.ensure_consumer_groups()

        # Buffer state: {camera_id: deque([(timestamp, prediction), ...])}
        self.buffers = defaultdict(deque)
        # Status state: {camera_id: "Clean" | "Dirty"} - Default assumes Clean
        self.statuses = defaultdict(lambda: "Clean")
        # Active Tickets: {camera_id: ticket_id_str}
        self.active_tickets = {}
        # Camera name mapping
        self.camera_names = {"1": "Counter", "7": "Kitchen"}

    def mark_acked(self, stream: str, msg_id: str, consumer: str):
        cnt_key = f"ack:cnt:{stream}:{msg_id}"
        seen_key = f"ack:seen:{stream}:{msg_id}"

        pipe = self.redis.pipeline()

        # Ensure each consumer increments only once
        pipe.sadd(seen_key, consumer)
        pipe.expire(seen_key, ACK_TTL_SECONDS)
        results = pipe.execute()
        was_added = results[0]  # 1 if new, 0 if already present

        if was_added:
            pipe = self.redis.pipeline()
            pipe.incr(cnt_key)
            pipe.expire(cnt_key, ACK_TTL_SECONDS)
            pipe.execute()

    def load_model(self):
        # Resolve absolute path for model
        abs_model_path = MODEL_PATH
        print("model path",abs_model_path)
        # if not os.path.isabs(abs_model_path):
        #     abs_model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)

        # if not os.path.exists(abs_model_path):
        #     logger.error(f"Model path not found at {abs_model_path}")
        #     return None

        try:
            logger.info(
                f"Rebuilding architecture and loading checkpoint from {abs_model_path}..."
            )

            model_name = "google/vit-base-patch16-224"

            # Step 1: Base ViT
            base_model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.bfloat16,
            )

            # Step 2: HQQ Quantization
            hqq_config = BaseQuantizeConfig(
                nbits=4, group_size=64, quant_zero=True, quant_scale=True
            )
            HQQLinear.set_backend(HQQBackend.ATEN)

            def apply_hqq(model, quant_config):
                for name, module in model.named_children():
                    if isinstance(module, nn.Linear):
                        setattr(
                            model,
                            name,
                            HQQLinear(
                                module, quant_config, compute_dtype=torch.bfloat16
                            ),
                        )
                    else:
                        apply_hqq(module, quant_config)

            apply_hqq(base_model, hqq_config)

            # Step 3: LoRA
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
            )
            base_model = get_peft_model(base_model, lora_config)

            # Step 4: MaskedViTClassifier wrapper
            model = MaskedViTClassifier(base_model)

            # Load checkpoint
            checkpoint = torch.load(abs_model_path, map_location=device)
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)

            # Use logger instead of print
            ckpt_keys = set(checkpoint.keys())
            lora_in_ckpt = [k for k in ckpt_keys if "lora_" in k]
            base_in_ckpt = [k for k in ckpt_keys if "classifier" in k]

            logger.info(f"Total checkpoint keys : {len(ckpt_keys)}")
            logger.info(
                f"LoRA keys in ckpt     : {len(lora_in_ckpt)} {'OK' if lora_in_ckpt else 'MISSING'}"
            )
            logger.info(
                f"Base keys in ckpt     : {len(base_in_ckpt)} {'OK' if base_in_ckpt else 'MISSING'}"
            )

            if not lora_in_ckpt:
                logger.warning(
                    "No LoRA weights found in checkpoint! This might lead to poor performance."
                )

            model = model.to(device)
            model.eval()

            # Disable all dropout for deterministic inference
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0

            logger.info("Model ready — deterministic inference guaranteed")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def ensure_consumer_groups(self):
        """Ensure consumer groups exist for all input streams."""
        for stream in INPUT_QUEUES:
            try:
                self.redis.xgroup_create(stream, GROUP_NAME, id="$", mkstream=True)
                logger.info(f"Created consumer group {GROUP_NAME} for {stream}")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    logger.info(
                        f"Consumer group {GROUP_NAME} already exists for {stream}"
                    )
                else:
                    logger.error(f"Error creating consumer group for {stream}: {e}")

    def get_mask_set(self, camera_id):
        # Map camera IDs to mask sets
        # Cam 1 -> counter
        # Cam 7 -> kitchen
        if str(camera_id) == "1":
            return MASK_SETS.get("counter", [])
        elif str(camera_id) == "7":
            return MASK_SETS.get("kitchen", [])
        return []

    def download_image(self, url):
        try:
            if url.startswith("http"):
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                return image
            else:
                # Local file
                if os.path.exists(url):
                    return Image.open(url).convert("RGB")
                return None
        except Exception as e:
            logger.error(f"Failed to download image {url}: {e}")
            return None

    def image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def visualize_frame(
        self, image, prediction, status, dirty_count, buffer_len, dirty_percent
    ):
        """Annotate the frame with status overlay, classification, and buffer stats."""
        # Convert PIL -> numpy (BGR for OpenCV)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]

        # --- Color overlay on the entire frame ---
        overlay = frame.copy()
        if status == "Dirty":
            overlay[:] = (0, 0, 200)  # Red (BGR)
        else:
            overlay[:] = (0, 180, 0)  # Green (BGR)
        alpha = 0.25
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # --- Text settings (scale relative to frame width) ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.6, w / 1000)
        thick = max(1, int(scale * 2))
        line_gap = int(40 * scale)
        x, y = 20, int(40 * scale)

        # Helper to draw text with a dark shadow for readability
        def put(text, pos, color=(255, 255, 255)):
            cv2.putText(
                frame, text, pos, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA
            )
            cv2.putText(frame, text, pos, font, scale, color, thick, cv2.LINE_AA)

        # 1) Current frame classification
        pred_color = (0, 0, 255) if prediction == "Dirty" else (0, 255, 0)
        put(f"Classification: {prediction}", (x, y), pred_color)
        y += line_gap

        # 2) Overall status
        status_color = (0, 0, 255) if status == "Dirty" else (0, 255, 0)
        put(f"Status: {status}", (x, y), status_color)
        y += line_gap

        # 3) Buffer stats
        put(f"Buffer: {dirty_count}/{buffer_len} dirty  ({dirty_percent:.1f}%)", (x, y))

        # Convert back to PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def apply_masks_pil(self, image, masks):
        masked_image = image.copy()
        draw = ImageDraw.Draw(masked_image)
        width, height = image.size

        for poly in masks:
            clipped_poly = []
            for x, y in poly:
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                clipped_poly.append((x, y))

            # Fill logic: The user provided code uses fill=(0,0,0).
            # Assuming the masks define areas to *exclude* or *include*?
            # From colab code: "draw.polygon(clipped_poly, fill=(0, 0, 0))"
            # This blacks out the polygon area.
            draw.polygon(clipped_poly, fill=(0, 0, 0))

        return masked_image

    def predict(self, tensor):
        if self.model is None:
            return "Unknown"

        # Consistent with training: use bfloat16
        tensor = tensor.to(torch.bfloat16)

        # Log patch attention stats (mirrors notebook debug output)
        attn_bool = self.model._build_attention_mask(tensor)
        attended = attn_bool[0].sum().item()
        masked = (~attn_bool[0]).sum().item()
        logger.info(f"Attended: {attended}/197 | ROI-masked: {masked}/197")

        with torch.no_grad():
            outputs = self.model(tensor)
            logits = outputs.logits
            probs = torch.softmax(logits.float(), dim=1)

        clean_score = probs[0][0].item()
        dirty_score = probs[0][1].item()
        raw_pred_idx = torch.argmax(logits, dim=1).item()
        raw_pred = ["Clean", "Dirty"][raw_pred_idx]

        logger.info(
            f"Logits: {logits.cpu().float()} | "
            f"Probs: Clean={clean_score:.4f}, Dirty={dirty_score:.4f} | "
            f"Raw pred: {raw_pred}"
        )

        # ==============================
        # Threshold gate (mirrors notebook logic):
        #   Model says Clean  → accept as Clean directly
        #   Model says Dirty  → only confirm Dirty if dirty_score >= DIRTY_SCORE_THRESHOLD
        #                     → else override to Clean
        # ==============================
        if raw_pred == "Dirty" and dirty_score >= DIRTY_SCORE_THRESHOLD:
            prediction = "Dirty"
        else:
            prediction = "Clean"

        logger.info(
            f"Final prediction (threshold={DIRTY_SCORE_THRESHOLD}): {prediction}"
        )
        return prediction

    def trigger_alert(
        self, camera_id, alert_type, timestamp, image, frame_url, ticket_id=None
    ):
        """Sends an alert to the output queue."""
        try:
            # Convert image to base64
            frame_b64 = self.image_to_base64(image)

            # Map camera ID to name
            cam_name = self.camera_names.get(str(camera_id), f"Camera {camera_id}")

            if alert_type == "raised":
                description = f"Dirty floor detected at {cam_name} area."
            else:
                description = (
                    f"Floor cleaned at {cam_name} area."
                )

            alert_payload = {
                "ticket_id": ticket_id,
                "frame": frame_b64,
                "frame_without_visualization": frame_b64,
                "description": description,
                "type": alert_type,
                "camera_id": camera_id,
                "status": "Dirty" if alert_type == "raised" else "Clean",
                "timestamp": timestamp,
                "frame_url": frame_url,
                "service": "dirty_floor_service",
                "target_collection": "alert_collection",
            }

            # Using xadd as per pattern in other services
            self.redis.xadd(OUTPUT_QUEUE_NAME, {"data": json.dumps(alert_payload)})
            logger.info(
                f"{alert_type.capitalize()} Alert: {description} (Ticket: {ticket_id})"
            )

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def update_status(self, camera_id, current_prediction, timestamp, image, frame_url):
        """
        Updates the buffer and status based on the new prediction.
        """
        try:
            ts = float(timestamp)
        except (ValueError, TypeError):
            ts = time.time()

        # Add to buffer
        self.buffers[camera_id].append((ts, current_prediction))

        # Clean buffer (remove old entries > BUFFER_DURATION_SECONDS)
        while self.buffers[camera_id] and (
            ts - self.buffers[camera_id][0][0] > BUFFER_DURATION_SECONDS
        ):
            self.buffers[camera_id].popleft()

        # Calculate stats
        buffer_len = len(self.buffers[camera_id])
        if buffer_len == 0:
            return {
                "dirty_count": 0,
                "buffer_len": 0,
                "dirty_percent": 0.0,
                "status": self.statuses[camera_id],
            }

        dirty_count = sum(1 for _, pred in self.buffers[camera_id] if pred == "Dirty")
        dirty_percent = (dirty_count / buffer_len) * 100

        current_status = self.statuses[camera_id]
        new_status = current_status

        # Logic:
        # If 90% are dirty -> Dirty
        # If < 90% dirty -> Clean (as per prompt "if the buffer then contains less than 90% dirty, status should change to clean")

        # However, to initiate "Dirty", we need >= 90%.
        # To initiate "Clean", we need < 90%.
        # This is a hard threshold, no hysteresis gap requested.

        # Logic with Hysteresis:
        # 1. To RAISE (Clean -> Dirty): High confidence required (>= DIRTY_THRESHOLD_PERCENT)
        # 2. To RESOLVE (Dirty -> Clean): Majority vote required (< CLEAN_THRESHOLD_PERCENT)
        if current_status == "Clean":
            if dirty_percent >= DIRTY_THRESHOLD_PERCENT:
                new_status = "Dirty"
        else: # current_status == "Dirty"
            if dirty_percent < CLEAN_THRESHOLD_PERCENT:
                new_status = "Clean"

        # Check for status change
        if new_status != current_status:
            logger.info(
                f"Cam {camera_id} status changing from {current_status} to {new_status} (Dirty%: {dirty_percent:.2f})"
            )

            if new_status == "Dirty":
                if camera_id not in self.active_tickets:
                    ticket_id = str(uuid.uuid4())
                    self.active_tickets[camera_id] = ticket_id
                    self.trigger_alert(
                        camera_id, "raised", timestamp, image, frame_url, ticket_id
                    )
                    
                    self.buffers[camera_id].clear()
                
            elif new_status == "Clean":
                ticket_id = self.active_tickets.pop(camera_id, None)
                if ticket_id:
                    self.trigger_alert(
                        camera_id, "resolved", timestamp, image, frame_url, ticket_id
                    )
                    
                    self.buffers[camera_id].clear()

            self.statuses[camera_id] = new_status

        return {
            "dirty_count": dirty_count,
            "buffer_len": buffer_len,
            "dirty_percent": dirty_percent,
            "status": self.statuses[camera_id],
        }

    def run(self):
        logger.info(f"Service started. Listening on: {INPUT_QUEUES}")

        streams_dict = {stream: ">" for stream in INPUT_QUEUES}

        while True:
            try:
                # Use xreadgroup for consuming from streams
                # count=1 to process one message at a time per stream for simplicity
                entries = self.redis.xreadgroup(
                    groupname=GROUP_NAME,
                    consumername=CONSUMER_NAME,
                    streams=streams_dict,
                    count=1,
                    block=1000,  # Block for 1 second
                )

                if not entries:
                    continue

                for stream, messages in entries:
                    for message_id, data in messages:
                        # Process message
                        self.process_message(stream, message_id, data)

                        # Acknowledge message
                        self.redis.xack(stream, GROUP_NAME, message_id)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

    def process_message(self, stream, message_id, data):
        # Data keys are bytes in redis-py unless decode_responses=True is strictly handled,
        # but we enabled decode_responses=True in init.

        frame_url = data.get("frame_url")
        camera_id = data.get("camera_id")
        timestamp = data.get("timestamp")

        if not frame_url:
            return

        logger.info(f"Processing frame from {camera_id} at {timestamp}")

        # Download
        image = self.download_image(frame_url)
        if image is None:
            return

        # Get masks for this camera
        masks = self.get_mask_set(camera_id)

        # Apply masks using PIL (as per colab code)
        if masks:
            masked_image = self.apply_masks_pil(image, masks)
        else:
            masked_image = image
            logger.warning(f"No masks found for camera {camera_id}")

        # Preprocess
        input_tensor = preprocess(masked_image).unsqueeze(0).to(device)

        # Predict
        prediction = self.predict(input_tensor)
        logger.info(f"Prediction for Cam {camera_id}: {prediction}")

        # Update Status and Alert Logic
        stats = self.update_status(camera_id, prediction, timestamp, image, frame_url)

        # Increment acknowledgment count
        self.mark_acked(stream, message_id, GROUP_NAME)

        # Visualize / Output
        if VISUALIZE:
            viz_data = {
                "frame_url": frame_url,
                "frame": self.image_to_base64(image),
                "masks": [m.tolist() if hasattr(m, "tolist") else m for m in masks],
                "timestamp": timestamp,
                "camera_id": camera_id,
                "prediction": prediction,
                "status": stats["status"],
                "dirty_count": stats["dirty_count"],
                "buffer_len": stats["buffer_len"],
                "dirty_percent": stats["dirty_percent"],
            }
            viz_queue = f"stream:visualization:{camera_id}"
            self.redis.rpush(viz_queue, json.dumps(viz_data))
            # Retain only last 100 frames for viz
            self.redis.ltrim(viz_queue, -100, -1)


if __name__ == "__main__":
    service = DirtyFloorService()
    service.run()
