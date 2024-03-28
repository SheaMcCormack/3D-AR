# AR Marker Detection and Calibration Script

This Markdown document explains the steps of the provided Python script for AR marker detection and calibration.

## Step 1: Load Calibration Sheet Image

The script loads the calibration sheet image ('calibration_sheet.jpg') in grayscale using the OpenCV library.

## Step 2: Feature Extraction (Shi-Tomasi Corner Detection)

- The Shi-Tomasi corner detection algorithm is employed to detect corners in the calibration sheet image.
- Key interest points (corners) are identified based on local intensity variations.
- The `goodFeaturesToTrack` function in OpenCV's `cv2` module is used to detect corners.
- Parameters such as the maximum number of corners (`maxCorners`), the minimum quality level (`qualityLevel`), and the minimum Euclidean distance between detected corners (`minDistance`) are specified.

## Step 3: Descriptor Generation (SIFT)

- The Scale-Invariant Feature Transform (SIFT) algorithm is initialized to compute descriptors for the detected corners.
- SIFT descriptors encode information about the local image regions surrounding each corner.
- These descriptors are invariant to image scale and rotation, making them suitable for matching keypoints across different views.

## Step 4: Marker Recognition

- Video is captured from the camera in real-time using the OpenCV `VideoCapture` object.
- Keypoints and descriptors are extracted from each frame of the camera feed using SIFT.
- Brute-force matching is performed between descriptors from the calibration sheet and the camera feed to identify potential marker matches.
- A ratio test is applied to filter out good matches.

## Step 5: Marker Detection and Pose Estimation

- If a sufficient number of good matches are found, the marker is detected.
- Source and destination points are extracted from the matched keypoints for homography estimation.
- The homography matrix is estimated using RANSAC.
- Perspective transformation is applied to align the corners of the calibration sheet with the detected marker in the camera frame.
- The marker outline is drawn on the camera frame using the transformed corners.

## Step 6: Display Marker Detection Result

- The camera frame with the marker outline is displayed using the OpenCV `imshow` function.
- If the marker is not detected, the original camera frame is displayed.

## Step 7: Calibration Process

- A calibration process is initiated to establish a correspondence between virtual and real-world coordinates.
- Known distances on the calibration sheet (in pixels) are measured.
- A calibration factor (`mm_per_pixel`) is calculated based on the measured distances.

## Step 8: Calibration Factor Calculation

- Dummy numbers are used in this example to demonstrate the calculation of the calibration factor (`mm_per_pixel`).
- The calibration factor represents the real-world distance (in mm) per pixel in the image.

## Conclusion

This Python script provides functionality for AR marker detection and calibration, allowing for accurate alignment of virtual objects with the real-world environment.

For any inquiries or issues, please contact [Your Name/Email].
