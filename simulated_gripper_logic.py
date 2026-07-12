import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

# This value controls how close the object needs to be to the center
# before we say it is ready to grip.
CENTER_TOLERANCE = 50

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Get frame size
    frame_height, frame_width, _ = frame.shape

    # Find the center of the camera frame
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    # Draw the frame center
    cv2.circle(frame, (frame_center_x, frame_center_y), 6, (0, 0, 255), -1)
    cv2.putText(
        frame,
        "Frame Center",
        (frame_center_x - 70, frame_center_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # Run YOLO detection
    results = model.predict(frame, conf=0.25, verbose=False)

    movement_instruction = "No object detected"

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Find object center
        object_center_x = int((x1 + x2) / 2)
        object_center_y = int((y1 + y2) / 2)

        # Calculate offset from frame center
        offset_x = object_center_x - frame_center_x
        offset_y = object_center_y - frame_center_y

        # Decide simulated gripper movement
        if abs(offset_x) <= CENTER_TOLERANCE and abs(offset_y) <= CENTER_TOLERANCE:
            movement_instruction = "Object centered - ready to grip"
        elif abs(offset_x) > abs(offset_y):
            if offset_x < 0:
                movement_instruction = "Move gripper left"
            else:
                movement_instruction = "Move gripper right"
        else:
            if offset_y < 0:
                movement_instruction = "Move gripper up"
            else:
                movement_instruction = "Move gripper down"

        # Draw object box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw object center
        cv2.circle(frame, (object_center_x, object_center_y), 6, (255, 0, 0), -1)

        # Draw line from frame center to object center
        cv2.line(
            frame,
            (frame_center_x, frame_center_y),
            (object_center_x, object_center_y),
            (255, 255, 255),
            2
        )

        label = f"{class_name} {confidence:.2f} | Offset: ({offset_x}, {offset_y})"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        print(
            f"Object: {class_name}, "
            f"Center: ({object_center_x}, {object_center_y}), "
            f"Offset: ({offset_x}, {offset_y}), "
            f"Instruction: {movement_instruction}"
        )

        # Only use the first detected object for now
        break

    # Show movement instruction on screen
    cv2.putText(
        frame,
        movement_instruction,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Simulated Gripper Logic", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()