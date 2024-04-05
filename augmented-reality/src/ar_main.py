
# Useful links
# http://www.pygame.org/wiki/OBJFileLoader
# https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
# https://clara.io/library

# TODO -> Implement command line arguments (scale, model and object to be projected)
#      -> Refactor and organize code (proper funcition definition and separation, classes, error handling...)

import argparse

import cv2
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

    cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        ret, frame = cap.read()
        print(frame.shape)
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

        # show result
        cv2.imshow('frame', dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    main()