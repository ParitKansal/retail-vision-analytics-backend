FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update -y && apt install ffmpeg -y
COPY service.py /workspace/service.py
COPY settings.py /workspace/settings.py
COPY .env /workspace/.env
COPY crew_room_mask.json /workspace/crew_room_mask.json
COPY counter_phone.json /workspace/counter_phone.json
COPY back_lobby_mask.json /workspace/back_lobby_mask.json
COPY counter_mask_floor_cleaning.json /workspace/counter_mask_floor_cleaning.json
COPY kiosk_mask.json /workspace/kiosk_mask.json
COPY left_lobby_mask.json /workspace/left_lobby_mask.json
COPY s3_manager.py /workspace/s3_manager.py

CMD ["python", "/workspace/service.py"]