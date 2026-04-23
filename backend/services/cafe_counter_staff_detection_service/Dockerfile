FROM python:3.10-slim

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first (layer cached unless requirements.txt changes)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (changes frequently)
COPY service.py .
COPY setting.json .

CMD ["python", "service.py"]