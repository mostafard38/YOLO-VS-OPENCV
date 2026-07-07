import cv2
from ultralytics import YOLO

# Load the YOLOv8 nano model.
# This model is small and fast, so it works well for webcam testing.
model = YOLO("yolov8n.pt")

# Open the default webcam.
# On MacBook, camera index 0 usually means the built-in webcam.
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully.
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

while True:
    # Read one frame from the webcam.
    ret, frame = cap.read()

    # If the frame was not read properly, stop the program.
    if not ret:
        print("Could not read frame.")
        break

    # Run YOLO object detection on the current webcam frame.
    results = model.predict(frame, conf=0.25, verbose=False)

    # Loop through all detected objects in the frame.
    for box in results[0].boxes:
        # Get the class ID and class name of the detected object.
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Get the confidence score of the detection.
        confidence = float(box.conf[0])

        # Get the bounding box coordinates.
        # x1, y1 = top-left corner
        # x2, y2 = bottom-right corner
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Convert coordinates to integers so OpenCV can draw them.
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Calculate the center point of the object.
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Draw the bounding box around the detected object.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw a small circle at the center of the object.
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Create text showing the object name, confidence, and coordinates.
        label = f"{class_name} {confidence:.2f} | Center: ({center_x}, {center_y})"

        # Display the label above the bounding box.
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Print the object information in the terminal.
        print(f"Object: {class_name}, Confidence: {confidence:.2f}, Center: x={center_x}, y={center_y}")

    # Show the webcam feed with boxes and coordinates.
    cv2.imshow("YOLO Object Coordinates", frame)

    # Press q to stop the program.
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()