import io
import torch
import requests
from PIL import Image, ImageDraw
from litserve import LitAPI, LitServer
from transformers import AutoProcessor, XCLIPModel, BitsAndBytesConfig
import os
from dotenv import load_dotenv
import logging
import logging_loki
from multiprocessing import Queue
import numpy as np

load_dotenv()

# Configure Logging
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

logger = logging.getLogger("xclip_service")
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
        tags={"application": "xclip_service"},
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


class XClipService(LitAPI):
    def setup(self, device=DEVICE):
        try:
            logger.info("Starting XClip service setup")
            self.device = device
            self.model_path = "/workspace/models/xclip-model"
            self.hf_model_id = "microsoft/xclip-base-patch16-zero-shot"
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # os.makedirs("/workspace/models/images", exist_ok=True)

            # Check if model exists locally
            if not os.path.exists(os.path.join(self.model_path, "config.json")):
                logger.info(
                    f"Model not found at {self.model_path}. Downloading from {self.hf_model_id}..."
                )

                # Ensure directory exists
                os.makedirs(self.model_path, exist_ok=True)

                # Download and save Processor
                processor = AutoProcessor.from_pretrained(self.hf_model_id)
                processor.save_pretrained(self.model_path)

                # Download and save Model
                model = XCLIPModel.from_pretrained(self.hf_model_id)
                model.save_pretrained(self.model_path)
                logger.info("Model downloaded and saved locally.")
            else:
                logger.info(f"Found local model at {self.model_path}")

            logger.info(f"Loading processor from {self.model_path}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )

            logger.info(f"Loading model from {self.model_path} with 4-bit quantization")
            self.model = XCLIPModel.from_pretrained(
                self.model_path,
                quantization_config=self.quant_config,
                local_files_only=True,
            ).to(self.device)
            self.model.eval()

            logger.info(
                f"{self.__class__.__name__} service setup complete on device: {self.device}"
            )
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}", exc_info=True)
            raise

    def decode_request(self, request):
        """Decode the request and return the image and text
        Args:
            request (dict): The request dictionary
            request is of the form {"texts": ["text1", "text2", ...], "image_urls": ["url1", "url2", ...]}
        Returns:
            tuple: The image and text
        """
        try:
            # print(request)
            texts = request.get("texts", "")
            images = []
            if "image_urls" in request:
                image_urls = request["image_urls"]
                logger.info(f"Received request with image URL: {image_urls[:3]}...")
                try:
                    for image_url in image_urls:
                        response = requests.get(image_url, timeout=10)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")
                        images.append(image)
                        logger.info(
                            "Successfully downloaded and processed image from URL"
                        )
                except Exception as e:
                    logger.error(f"Failed to fetch image from URL: {e}")
                    raise e
            else:
                logger.error("No image URLs provided in the request")
                raise ValueError("No image URLs provided in the request")
            if len(images) < 32:
                # insert black images to make it 32
                img_size = images[0].size
                for i in range(32 - len(images)):
                    images.append(Image.new("RGB", img_size, (0, 0, 0)))
            return images, texts
        except Exception as e:
            logger.error(f"Error in decode_request: {str(e)}", exc_info=True)
            raise

    def predict(self, inputs):
        try:
            images, texts = inputs
            logger.info(f"Running prediction with text prompt: {texts}")
            probs = []
            for text in texts:
                prompt = text.get("prompt")
                mask = text.get("mask")
                if mask:
                    masked_images = []

                    for img in images:
                        w, h = img.size

                        # Convert polygon safely inside bounds
                        polygon_mask = np.array(
                            [
                                [
                                    max(0, min(int(p["x"]), w - 1)),
                                    max(0, min(int(p["y"]), h - 1)),
                                ]
                                for p in mask
                            ],
                            dtype=np.int32,
                        )

                        mask_img = Image.new("L", (w, h), 0)
                        draw = ImageDraw.Draw(mask_img)
                        draw.polygon(polygon_mask.tolist(), fill=255)

                        masked = Image.composite(
                            img,
                            Image.new("RGB", (w, h), (0, 0, 0)),
                            mask_img,
                        )

                        masked_images.append(masked)

                    images_for_model = masked_images
                else:
                    images_for_model = images

                # for i in range(len(images_for_model)):
                #     images_for_model[i].save(f"/workspace/models/images/image_{i}.jpg")

                text_inputs = self.processor.tokenizer(
                    prompt, return_tensors="pt", padding=True
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                video_inputs = self.processor(
                    images=images_for_model, return_tensors="pt"
                ).to(self.device)
                pixel_key = list(video_inputs.keys())[0]
                pixel_values = video_inputs[pixel_key]
                if pixel_values.dim() == 4:
                    pixel_values = pixel_values.unsqueeze(0)

                inputs = {"pixel_values": pixel_values.to(self.device), **text_inputs}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_video = outputs.logits_per_video
                    probabilities = logits_per_video.softmax(dim=-1)
                    probs.append(probabilities.squeeze().detach().cpu().tolist())

            results = {"probs": probs, "texts": texts}
            logger.info(f"Prediction successful. Found {len(probs)} detections")
            print(results)
            return results

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise

    def encode_response(self, prediction):
        try:
            logger.info("Encoding response")
            return prediction
        except Exception as e:
            logger.error(f"Error encoding response: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    server = LitServer(XClipService(), accelerator="auto")
    port = int(os.getenv("PORT", "8085"))
    logger.info(f"Starting server on port {port}")
    server.run(port=port)
