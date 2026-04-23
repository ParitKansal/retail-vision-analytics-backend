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

# Copy essential service files
COPY service.py /workspace/service.py
COPY detector.py /workspace/detector.py
COPY config.py /workspace/config.py
COPY regions.py /workspace/regions.py
COPY geometry_utils.py /workspace/geometry_utils.py
COPY setting.json /workspace/setting.json

CMD [ "python" ,"service.py"]
