import cv2
import time
from ultralytics import YOLO


CAMERA_INDEX = 0
NUM_FRAMES = 100


def test_opencv_latency():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Could not open webcam for OpenCV test.")
        return None

    total_time = 0

    print("Running OpenCV latency test...")

    for _ in range(NUM_FRAMES):
        start_time = time.time()

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        cv2.imshow("OpenCV Only", frame)

        end_time = time.time()
        total_time += end_time - start_time

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    average_delay = total_time / NUM_FRAMES
    return average_delay


def test_yolo_latency():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Could not open webcam for YOLO test.")
        return None

    total_time = 0

    print("Running YOLO latency test...")

    for _ in range(NUM_FRAMES):
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        start_time = time.time()

        results = model.predict(frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Detection", annotated_frame)

        end_time = time.time()
        total_time += end_time - start_time

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    average_delay = total_time / NUM_FRAMES
    return average_delay


opencv_delay = test_opencv_latency()
yolo_delay = test_yolo_latency()

if opencv_delay is not None and yolo_delay is not None:
    difference = yolo_delay - opencv_delay

    print("\n--- Latency Results ---")
    print(f"OpenCV average delay per frame: {opencv_delay:.4f} seconds")
    print(f"YOLO average delay per frame: {yolo_delay:.4f} seconds")
    print(f"Difference: {difference:.4f} seconds")
    print(f"OpenCV FPS: {1 / opencv_delay:.2f}")
    print(f"YOLO FPS: {1 / yolo_delay:.2f}")