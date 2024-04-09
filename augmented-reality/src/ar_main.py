
import argparse

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import os
from objloader_simple import *

def main():
    # camera calibration matrix
    cameraMatrix = np.load('../calibration/camera_matrix.npy')#np.array([[655.24548568, 0.0, 313.17837698], [0.0, 659.32398974, 244.03682075], [0.0, 0.0, 1.0]])
    dist = np.load('../calibration/dist_coeffs.npy')#np.array([[-0.41837736, 0.24344121, -0.00069054,  0.00109116, -0.34367113]])
    
    # Load 3D model from OBJ file
    dir_name = os.getcwd()
    obj = OBJ(os.path.join(dir_name, 'models/pirate-ship-fat.obj'), swapyz=True)

    # Initialize the Charuco board and detector
    ARUCO_DICT = cv2.aruco.DICT_6X6_250
    SQUARES_VERTICALLY = 7
    SQUARES_HORIZONTALLY = 5
    SQUARE_LENGTH = 0.03
    MARKER_LENGTH = 0.015
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Generate and save the AR marker image
    #marker_id = 99
    #marker_size = 400
    #marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
    #cv2.imwrite("marker_image.png", marker_image)

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
        undistorted_image = dst.copy()

        # Detect markers in the undistorted image
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(dst, dictionary, parameters=params)

        # draw detected markers
        if marker_ids is not None and len(marker_ids) >= 6:
            # Interpolate CharUco corners
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, dst, board)

            # If enough corners are found, estimate the pose
            if charuco_retval:
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, cameraMatrix, dist, None, None)

                # If pose estimation is successful, draw the axis
                if retval:
                    cv2.drawFrameAxes(undistorted_image, cameraMatrix, dist, rvec, tvec, length=0.1, thickness=15)
                    
            #undistorted_image = cv2.aruco.drawDetectedCornersCharuco(undistorted_image, charuco_corners)
            if False:
                src_pts = np.array([[0, 0], [0, marker_size], [marker_size, marker_size], [marker_size, 0]], dtype=np.float32)
                dst_pts = np.array(corners[0][0], dtype=np.float32)

                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)   
                homography *= -1
                
                # Warp the marker image based on the homography matrix
                warped_image = cv2.warpPerspective(marker_image, homography, (dst.shape[1], dst.shape[0]))
                warped_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
                
                # Create a mask of the warped image
                mask = np.zeros_like(dst)
                mask = cv2.fillConvexPoly(mask, np.int32(dst_pts), (255,)*dst.shape[2])

                # Overlay the warped image onto the original frame
                dst = cv2.bitwise_and(dst, cv2.bitwise_not(mask))
                dst = cv2.add(dst, warped_image)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(cameraMatrix, homography)  
                        # project cube or model
                        dst = render(dst, obj, projection, h, w)
                    except:
                        pass

        # show result
        cv2.imshow('frame', undistorted_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, h, w, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 100

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (0, 0, 100))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

if __name__ == '__main__':
    main()