FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update
# RUN apt-get update && apt-get install -y \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
COPY service.py /workspace/service.py

EXPOSE 8085

CMD [ "python" ,"service.py"]