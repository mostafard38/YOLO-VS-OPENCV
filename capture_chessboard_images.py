import cv2
import os

# Folder where the chessboard images will be saved
SAVE_FOLDER = "calibration_images"

# Create the folder if it does not already exist
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Open the default webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

image_count = 0

print("Press 's' to save a chessboard image.")
print("Press 'q' to quit.")

while True:
    # Read one frame from the webcam
    ret, frame = cap.read()

    # Stop if the camera frame cannot be read
    if not ret:
        print("Could not read frame.")
        break

    # Show the live webcam feed
    cv2.imshow("Capture Chessboard Images", frame)

    # Check which key is pressed
    key = cv2.waitKey(1)

    # Press 's' to save the current frame
    if key == ord("s"):
        image_path = os.path.join(SAVE_FOLDER, f"chessboard_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved: {image_path}")
        image_count += 1

    # Press 'q' to quit
    if key == ord("q"):
        break

# Close webcam and windows
cap.release()
cv2.destroyAllWindows()