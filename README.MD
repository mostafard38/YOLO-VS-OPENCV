# YOLO vs OpenCV myCobot Gripper Evaluation

This project explores YOLO and OpenCV for object detection and gripper evaluation using a myCobot robot arm.

## Purpose

The goal is to compare the delay time between YOLO-based detection and OpenCV-based detection during a robotic pick-and-grip task.

The experiment focuses on measuring:

1. Camera frame capture time
2. Object detection/fixation time
3. Robot command time
4. Gripper movement/pickup completion time

## Files

- `tasks.py` - YOLO webcam detection/tracking
- `camera_test.py` - OpenCV webcam test
- `latency_compare.py` - compares delay between OpenCV and YOLO
- `autolabelling.py` - auto-labels images using YOLO/SAM
- `callback_test.py` - YOLO callback example
- `export_callback.py` - YOLO export callback example

## Current Progress

- Installed and tested Ultralytics YOLO
- Confirmed YOLO works with MacBook webcam
- Tested OpenCV webcam capture
- Ran YOLO object detection and tracking
- Created auto-labels for sample images
- Started latency comparison between YOLO and OpenCV

## Main Research Question

Which vision approach introduces more delay during the myCobot gripper task: YOLO or OpenCV?

YOLO is expected to have higher perception delay because it performs AI inference on each frame. OpenCV is expected to be faster, but may be less flexible depending on object appearance, lighting, and background.