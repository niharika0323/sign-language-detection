import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import os

# Load the trained YOLO model
model = YOLO("C:/Users/ALOK/OneDrive/Desktop/best.pt")

print("Model loaded successfully!")

def speak_text(text):
    """
    Converts text to speech using gTTS and plays it.
    """
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        os.system("start output.mp3")
    except Exception as e:
        print(f"Error in Text-to-Speech: {e}")

def draw_detections(frame, results):
    """
    Draws bounding boxes, labels, and confidence scores on the frame.
    """
    detected_labels = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Ensure tensors are converted to Python scalars
            confidence = float(box.conf[0])  # Convert to Python float
            label = model.names[int(box.cls[0])]  # Convert to Python int for indexing
            detected_labels.append(label)

            # Draw rectangle around the detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label and confidence score
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_labels

def main_application():
    """
    Main application for real-time sign language detection and interpretation.
    """
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    print("Starting real-time detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from the camera.")
            break

        # Perform predictions
        results = model.predict(source=frame, conf=0.3, verbose=False)

        # Draw detections and collect recognized labels
        frame, detected_labels = draw_detections(frame, results)

        # Display the recognized labels as text
        if detected_labels:
            recognized_text = ", ".join(detected_labels)
            cv2.putText(frame, f"Detected: {recognized_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Convert detected labels to speech
            speak_text(recognized_text)

        # Show the frame
        cv2.imshow("Sign Language Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_application()