import cv2
import numpy as np
import time
import csv
from ultralytics import YOLO


CAMERA_INDEX = 0
NUM_FRAMES = 1000

model = YOLO("yolov8n.pt")


def detect_blue_with_opencv(frame):
    start_time = time.perf_counter()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_center = None
    bounding_box = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2

            object_center = (center_x, center_y)
            bounding_box = (x, y, w, h)

    end_time = time.perf_counter()
    delay = end_time - start_time

    return object_center, bounding_box, delay, mask


def detect_cellphone_with_yolo(frame):
    start_time = time.perf_counter()

    results = model.predict(frame, conf=0.25, verbose=False)

    object_center = None
    bounding_box = None
    class_name = None

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        detected_class = model.names[class_id]

        if detected_class == "cell phone":
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            object_center = (center_x, center_y)
            bounding_box = (int(x1), int(y1), int(x2), int(y2))
            class_name = detected_class
            break

    end_time = time.perf_counter()
    delay = end_time - start_time

    return object_center, bounding_box, class_name, delay


cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

opencv_times = []
yolo_times = []

with open("blue_object_latency_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "frame",
        "opencv_detected",
        "opencv_delay_seconds",
        "yolo_detected",
        "yolo_delay_seconds",
        "yolo_class"
    ])

    print("Starting blue object vs YOLO cell phone latency comparison...")
    print("Hold the blue phone/object in front of the camera.")
    print("Press q to stop early.")

    for frame_number in range(NUM_FRAMES):
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        opencv_center, opencv_box, opencv_delay, mask = detect_blue_with_opencv(frame)
        yolo_center, yolo_box, yolo_class, yolo_delay = detect_cellphone_with_yolo(frame)

        opencv_detected = opencv_center is not None
        yolo_detected = yolo_center is not None

        opencv_times.append(opencv_delay)
        yolo_times.append(yolo_delay)

        writer.writerow([
            frame_number,
            opencv_detected,
            opencv_delay,
            yolo_detected,
            yolo_delay,
            yolo_class
        ])

        display_frame = frame.copy()

        if opencv_box is not None:
            x, y, w, h = opencv_box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(display_frame, opencv_center, 5, (0, 255, 0), -1)
            cv2.putText(
                display_frame,
                "OpenCV blue object",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if yolo_box is not None:
            x1, y1, x2, y2 = yolo_box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(display_frame, yolo_center, 5, (255, 0, 0), -1)
            cv2.putText(
                display_frame,
                "YOLO cell phone",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        cv2.imshow("YOLO vs OpenCV Blue Object Latency", display_frame)
        cv2.imshow("OpenCV Blue Mask", mask)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

average_opencv = sum(opencv_times) / len(opencv_times)
average_yolo = sum(yolo_times) / len(yolo_times)
difference = average_yolo - average_opencv

print("\n--- Latency Results ---")
print(f"OpenCV average detection delay: {average_opencv:.6f} seconds")
print(f"YOLO average detection delay: {average_yolo:.6f} seconds")
print(f"Difference: {difference:.6f} seconds")
print(f"OpenCV FPS estimate: {1 / average_opencv:.2f}")
print(f"YOLO FPS estimate: {1 / average_yolo:.2f}")
print("\nResults saved to blue_object_latency_results.csv")