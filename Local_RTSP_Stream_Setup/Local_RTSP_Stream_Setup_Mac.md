## 🎯 Goal

Stream local videos (e.g.
`output_17_000.mp4` and `counter_2025-10-27_23-19-03.mp4`)
as **live looping RTSP feeds** on your Mac using **FFmpeg + MediaMTX**, and view them using **OpenCV** or **VLC**.



## ⚙️ Steps

### 1️⃣ Install

```bash
brew install ffmpeg mediamtx
``` 



### 2️⃣ Create MediaMTX config

```bash
mkdir -p ~/.config/mediamtx
cat > ~/.config/mediamtx/mediamtx.yml <<'YAML'
rtsp:
  udpDisable: yes   # Force TCP for stability
paths:
  all_others: {}    # Allow all paths
YAML
```



### 3️⃣ Start MediaMTX (Terminal 1)

```bash
mediamtx ~/.config/mediamtx/mediamtx.yml
```

Keep this running.
✅ You should see:

```
[RTSP] listener opened on :8554
[RTMP] listener opened on :1935
```



### 4️⃣ Stream Video 1 (Terminal 2)

```bash
ffmpeg -re -stream_loop -1 -i ~/Downloads/output_17_000.mp4 \
  -c:v h264_videotoolbox -b:v 1200k -g 50 -keyint_min 50 \
  -c:a aac -ar 44100 -b:a 128k \
  -f flv rtmp://127.0.0.1:1935/live/stream1
```



### 5️⃣ Stream Video 2 (Terminal 3)

```bash
ffmpeg -re -stream_loop -1 -i ~/Downloads/counter_2025-10-27_23-19-03.mp4 \
  -c:v h264_videotoolbox -b:v 1200k -g 50 -keyint_min 50 \
  -c:a aac -ar 44100 -b:a 128k \
  -f flv rtmp://127.0.0.1:1935/live/stream2
```



### 6️⃣ Test Locally

```bash
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/live/stream1
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/live/stream2
```



### 7️⃣ View on Other Devices (same Wi-Fi)

Find your IP:

```bash
ipconfig getifaddr en0
```

Example:

```
192.168.2.93
```

Then open in VLC or app:

```
rtsp://192.168.2.93:8554/live/stream1
rtsp://192.168.2.93:8554/live/stream2
```



### 8️⃣ View via Python (OpenCV)

```python
import cv2

# create a VideoCapture object for your RTSP stream
cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/live/stream2")

# check if the stream opened successfully
if not cap.isOpened():
    print("Cannot open RTSP stream")
else:
    print("RTSP stream opened successfully")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # show the live frame
        cv2.imshow("Live Stream", frame)
        
        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# cleanup
cap.release()
cv2.destroyAllWindows()
```

✅ Works for both `stream1` and `stream2` — just replace the URL.



### 9️⃣ Stop Everything

* Stop FFmpeg (each terminal): `Ctrl + C`
* Stop MediaMTX: `Ctrl + C`
* Verify ports closed:

  ```bash
  lsof -iTCP -sTCP:LISTEN -n -P | grep -E "8554|1935"
  ```
