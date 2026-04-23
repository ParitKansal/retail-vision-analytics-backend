import cv2
import os
from dotenv import load_dotenv
from service import CounterStaffDetector

load_dotenv()

# Get RTSP URL from environment variable or use a default/placeholder
# You can set RTSP_URL in your .env file
RTSP_URL = os.getenv("RTSP_URL")

def main():
    if not RTSP_URL:
        print("Error: RTSP_URL environment variable is not set.")
        print("Please set RTSP_URL in your .env file or export it.")
        return

    print(f"Connecting to RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    # Initialize the detector
    detector = CounterStaffDetector()
    print("Detector initialized. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from stream.")
            break

        # Preprocessing
        frame = detector.preprocessing(frame)

        # Run YOLOv8 inference
        results = detector.model(frame)

        # Post-processing (filtering by polygons)
        valid_boxes, pixel_polygons = detector.post_processing(results, frame.shape)

        # Visualization
        # Draw polygons
        for poly in pixel_polygons:
            cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

        # Draw valid detections
        for box in valid_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Staff", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("RTSP Test - Counter Staff Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
