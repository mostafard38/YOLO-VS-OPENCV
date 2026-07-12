import cv2
import numpy as np
import glob

# Number of INNER corners in your chessboard.
# If your board has 9 squares across and 7 squares down,
# then the inner corners are 8 across and 6 down.
CHESSBOARD_SIZE = (8, 6)

# Size of one square. Since you are using a phone screen,
# this is approximate. For learning, 1.0 is fine.
SQUARE_SIZE = 1.0

# Prepare known 3D points for the chessboard corners
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Lists to store real-world points and image points
object_points = []
image_points = []

# Load saved chessboard images
images = glob.glob("calibration_images/*.jpg")

print(f"Found {len(images)} calibration images.")

if len(images) == 0:
    print("No images found. Check that your images are inside the calibration_images folder.")
    exit()

gray = None

for image_path in images:
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read image: {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try to find chessboard inner corners
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        print(f"Detected chessboard corners in: {image_path}")

        object_points.append(objp)

        # Refine corner positions for better accuracy
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        image_points.append(refined_corners)

        # Draw detected corners so you can visually confirm it worked
        cv2.drawChessboardCorners(image, CHESSBOARD_SIZE, refined_corners, found)
        cv2.imshow("Detected Chessboard Corners", image)
        cv2.waitKey(500)

    else:
        print(f"Could NOT detect chessboard corners in: {image_path}")

cv2.destroyAllWindows()

if len(object_points) == 0:
    print("\nNo valid chessboard detections found.")
    print("Try taking clearer images with the full chessboard visible.")
    print("Also check if CHESSBOARD_SIZE matches your pattern.")
    exit()

# Calibrate camera
success, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    gray.shape[::-1],
    None,
    None
)

print("\n--- Camera Calibration Results ---")
print("Calibration success value:")
print(success)

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(distortion_coefficients)

# Save results
np.savez(
    "camera_calibration_results.npz",
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients
)

print("\nSaved calibration results to camera_calibration_results.npz")