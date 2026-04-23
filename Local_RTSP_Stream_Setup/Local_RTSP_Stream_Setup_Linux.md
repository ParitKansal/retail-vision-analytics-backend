
### ✅ **Steps You Completed Correctly on Linux**

1. **Installed FFmpeg**

   ```bash
   sudo apt update
   sudo apt install ffmpeg -y
   ```

2. **Verified FFmpeg installation**

   ```bash
   ffmpeg -version
   ```

3. **Created MediaMTX configuration directory**

   ```bash
   mkdir -p ~/.config/mediamtx
   ```

4. **Created MediaMTX config file**

   ```bash
   cat > ~/.config/mediamtx/mediamtx.yml <<'YAML'
   rtsp: yes
   rtmp: yes
   hls: yes
   webrtc: no
   paths:
     all_others: {}
   YAML
   ```

5. **Started MediaMTX**

   ```bash
   mediamtx ~/.config/mediamtx/mediamtx.yml
   ```

6. **Verified local IP address**

   ```bash
   ipconfig getifaddr en0
   ```

   *(or on Linux, equivalent command)*

   ```bash
   hostname -I
   ```

   → Example output: `192.168.2.93`

7. **Tested streaming a video using FFmpeg**

   ```bash
   ffmpeg -re -stream_loop -1 -i ~/McD/Table_Cleaning_Vids/output_17_000.mp4 \
     -c:v libx264 -preset veryfast -g 50 -keyint_min 50 -b:v 1200k \
     -c:a aac -ar 44100 -b:a 128k \
     -f flv rtmp://127.0.0.1:1935/live/stream1
   ```

8. **Opened stream using OpenCV (Python)**

   ```python
   import cv2

   cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/live/stream1")

   if not cap.isOpened():
       print("Cannot open RTSP stream")
   else:
       print("RTSP stream opened successfully")

       while True:
           ret, frame = cap.read()
           if not ret:
               print("Failed to grab frame")
               break

           cv2.imshow("Live Stream", frame)

           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

   cap.release()
   cv2.destroyAllWindows()
   ```

