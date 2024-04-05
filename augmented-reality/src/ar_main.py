
import argparse

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import os
from objloader_simple import *

def main():
    # camera calibration matrix
    cameraMatrix = np.array([[655.24548568, 0.0, 313.17837698], [0.0, 659.32398974, 244.03682075], [0.0, 0.0, 1.0]])
    dist = np.array([[-0.41837736, 0.24344121, -0.00069054,  0.00109116, -0.34367113]])
    
    # Load 3D model from OBJ file
    dir_name = os.getcwd()
    obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    # Generate and save the AR marker image
    marker_id = 99
    marker_size = 400
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
    cv2.imwrite("marker_image.png", marker_image)

    cap = cv2.VideoCapture(0)
    

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return 
        h,  w = frame.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

        # Undistort
        dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # detect markers
        corners, ids, _ = detector.detectMarkers(dst)

        # draw detected markers
        if ids is not None:
            aruco.drawDetectedMarkers(dst, corners)
            print(corners)

        # show result
        cv2.imshow('frame', dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    main()