FROM python:3.12-slim

WORKDIR /workspace

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTORCH_NUM_THREADS=1

# Install system dependencies (for cv2 if needed later, and general build utils)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY service.py /workspace/service.py

# Expecting the model file to be mounted or copied here
# COPY resnet50.pth /workspace/resnet50.pth 

EXPOSE 8000

CMD ["python", "service.py"]
