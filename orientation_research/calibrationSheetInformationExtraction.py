import cv2
import numpy as np

# Load calibration sheet image
calibration_sheet = cv2.imread('calibration_sheet.jpg', cv2.IMREAD_GRAYSCALE)

# Feature Extraction (Shi-Tomasi Corner Detection):
# Detect corners using the Shi-Tomasi corner detection algorithm.
# Shi-Tomasi corner detection identifies key interest points (corners) in the calibration sheet image
# based on local intensity variations in different directions.
# The `goodFeaturesToTrack` function in OpenCV's `cv2` module is used,
# specifying parameters such as the maximum number of corners to detect (`maxCorners`),
# the minimum quality level for corners (`qualityLevel`),
# and the minimum Euclidean distance between detected corners (`minDistance`).
corners = cv2.goodFeaturesToTrack(calibration_sheet, maxCorners=100, qualityLevel=0.01, minDistance=10)

# Descriptor Generation (SIFT):
# Initialize the SIFT (Scale-Invariant Feature Transform) algorithm.
# SIFT descriptors are computed for the detected corners,
# encoding information about the local image regions surrounding each corner.
# These descriptors are invariant to image scale and rotation,
# making them suitable for matching keypoints across different views.
# The `SIFT_create` function in OpenCV's `cv2` module initializes the SIFT algorithm,
# and the `compute` method of the SIFT object computes descriptors for the provided keypoints.
sift = cv2.SIFT_create()
keypoints, descriptors = sift.compute(calibration_sheet, corners)

# Marker Recognition
# Capture video from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and compute descriptors
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)
    
    # Match descriptors
    # Initialize a Brute-Force Matcher object (`bf`) using OpenCV's `cv2` module.
    # Then, use the `knnMatch` method of the matcher object (`bf`) to find the two best matches
    # for each descriptor in the first set (`descriptors`) compared to descriptors from the second set (`descriptors_frame`).
    # The parameter `k=2` specifies that we want to find the two best matches for each descriptor.
    # The `matches` variable will contain a list of matches, where each match is represented as a list
    # of two nearest neighbors for each descriptor in the first set (`descriptors`).
    # The first match in each list is the best match, and the second match is the second-best match.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors_frame, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # If enough matches are found, marker is detected
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Pose estimation
        h, w = calibration_sheet.shape
        corners_frame = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners_frame, H)
        
        # Draw marker outline
        frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display marker detection result
        cv2.imshow('Marker Detection', frame)
    else:
        cv2.imshow('Marker Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calibration Process
# Measure known distance on the calibration sheet (in pixels)
# Assume known distance between two corners is 100 pixels

# Calculate scale factor
#dummy numbers for now
known_distance_mm = 200  # Example: distance between two corners in real-world (in mm)
known_distance_pixels = 100  # Example: distance between two corners in image (in pixels)

# Calibration factor: mm_per_pixel
mm_per_pixel = known_distance_mm / known_distance_pixels

print("Calibration factor (mm per pixel):", mm_per_pixel)
