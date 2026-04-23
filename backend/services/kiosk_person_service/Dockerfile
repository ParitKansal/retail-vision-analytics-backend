FROM python:3.12-slim

WORKDIR /workspace

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
COPY service.py /workspace/service.py
COPY artifacts/settings.json /workspace/artifacts/settings.json

CMD [ "python" ,"service.py"]
