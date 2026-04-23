import os
import cv2

CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "rtsp://CAMERA_USER:CAMERA_PASS@CAMERA_IP:554/live")

cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print("❌ Cannot connect to RTSP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    cv2.imshow("RTSP Live Stream", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
