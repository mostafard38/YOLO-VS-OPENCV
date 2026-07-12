# YOLO vs OpenCV myCobot Gripper Evaluation

This project explores YOLO and OpenCV for object detection, object coordinates, and basic gripper movement logic.

The long-term goal is to build a simple vision pipeline that can detect an object, find its position in the camera frame, and decide how a robotic gripper should move toward it.

## Purpose

Before a robotic arm can pick up an object, it needs to understand what the camera is seeing.

This project focuses on:

1. Detecting objects using YOLO
2. Detecting a blue object using OpenCV color filtering
3. Comparing YOLO and OpenCV detection delay
4. Finding the center coordinates of detected objects
5. Testing camera calibration using a chessboard pattern
6. Simulating basic gripper movement decisions

## Main Files

- `camera_test.py` - basic OpenCV webcam test
- `tasks.py` - YOLO webcam detection/tracking test
- `coordinate_extractor.py` - detects objects and finds their center coordinates
- `blue_object_latency_compare.py` - compares detection delay between OpenCV and YOLO
- `capture_chessboard_images.py` - captures chessboard images for calibration
- `camera_calibration.py` - calculates camera matrix and distortion coefficients
- `simulated_gripper_logic.py` - uses object coordinates to simulate gripper movement
- `autolabelling.py` - auto-labels sample images using YOLO/SAM
- `callback_test.py` - YOLO callback example

## Current Progress

Completed:

- Installed and tested Ultralytics YOLO
- Confirmed YOLO works with the MacBook webcam
- Tested OpenCV webcam capture
- Ran YOLO object detection and tracking
- Created auto-labels for sample images
- Compared OpenCV and YOLO detection delay
- Extracted object center coordinates from YOLO bounding boxes
- Tested chessboard camera calibration
- Added simulated gripper movement logic

## Coordinate Extraction

YOLO returns a bounding box around each detected object.

The center of the object is calculated using:

```python
center_x = int((x1 + x2) / 2)
center_y = int((y1 + y2) / 2)

These are image coordinates, meaning they describe where the object is located in the camera frame.

## YOLO vs OpenCV Delay Comparison

OpenCV detects a blue object using HSV color filtering.

YOLO detects objects using a trained object detection model.

In my test, OpenCV was faster because it only looks for a color range. YOLO was slower, but it is more flexible because it can recognize different object classes.

## Simulated Gripper Logic

The simulated gripper script compares the object center to the center of the camera frame.

Example logic:

```text
Object left of center  -> move gripper left
Object right of center -> move gripper right
Object above center    -> move gripper up
Object below center    -> move gripper down
Object near center     -> ready to grip