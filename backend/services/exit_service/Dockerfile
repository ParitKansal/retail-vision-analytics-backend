FROM python:3.12-slim

WORKDIR /workspace

# Install system dependencies required for OpenCV and wget
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --default-timeout=1000 -r /workspace/requirements.txt

COPY service.py /workspace/service.py
COPY setting.json /workspace/setting.json
COPY face_detection_06022026.pt /workspace/face_detection_06022026.pt
COPY bnb_4bit_model /workspace/bnb_4bit_model
# Download yolo11n.pt if not present in context
RUN if [ ! -f /workspace/yolo11n.pt ]; then wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O /workspace/yolo11n.pt; fi
COPY emotion-ferplus-8.onnx /workspace/emotion-ferplus-8.onnx

ENV YOLO_CONFIG_DIR=/workspace/Ultralytics
ENV PYTHONUNBUFFERED=1

CMD [ "python", "-u", "service.py"]