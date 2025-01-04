import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load the YOLO model
model = YOLO("C:/Users/ALOK/OneDrive/Desktop/best.pt")
print("Model loaded successfully!")

def draw_detections(frame, results):
    """
    Draws bounding boxes, labels, and confidence scores on the frame.
    """
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get bounding box coordinates
        confidence = result.conf[0]  # Confidence score
        cls = int(result.cls[0])  # Class index
        label = model.names[cls]  # Get class name from model
        # Draw rectangle around the detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add label and confidence score
        text = f"{label} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Test OpenCV compatibility
def check_imshow():
    try:
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except Exception as e:
        print("WARNING: cv2.imshow() is not supported.")
        return False

if not check_imshow():
    raise EnvironmentError("cv2.imshow() is not available.")

# Real-time detection from webcam
def detect_from_camera():
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    print("Starting real-time detection. Press 'q' to quit.")
    fps_start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from the camera.")
            break
        
        # Perform predictions
        results = model.predict(source=frame, conf=0.3, verbose=False)[0]  # Disable verbose logging

        # Draw detections on the frame
        frame = draw_detections(frame, results)

        # Display FPS on the frame
        frame_count += 1
        fps_end_time = time.time()
        fps = frame_count / (fps_end_time - fps_start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show the frame
        cv2.imshow("Sign Language Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

# Start detection
detect_from_camera()
