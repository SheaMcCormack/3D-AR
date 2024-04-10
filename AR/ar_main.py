
import argparse

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import os
from objloader_simple import *

def main():
    # Camera calibration matrix
    cameraMatrix = np.load('../calibration/camera_matrix.npy')#np.array([[655.24548568, 0.0, 313.17837698], [0.0, 659.32398974, 244.03682075], [0.0, 0.0, 1.0]])
    dist = np.load('../calibration/dist_coeffs.npy')#np.array([[-0.41837736, 0.24344121, -0.00069054,  0.00109116, -0.34367113]])
    
    # Load 3D model from OBJ file
    dir_name = os.getcwd()
    obj = OBJ(os.path.join(dir_name, 'wolf.obj'), swapyz=True)
    
    # Initialize the detector
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    # Generate the AR marker image
    marker_id = 99
    marker_size = 400
    #marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
    #cv2.imwrite("marker_image.png", marker_image)
    base_src = np.array([[0, 0], [marker_size, 0], [marker_size, marker_size], [0, marker_size]], dtype=np.float32)
    marker_dict = {'69': {'src_pts': base_src}, 
                   '97': {'src_pts': base_src + np.array([[2*marker_size, 0], [2*marker_size, 0], [2*marker_size, 0], [2*marker_size, 0]], dtype=np.float32)}, 
                   '22':{'src_pts': base_src + np.array([[0, 2*marker_size], [0, 2*marker_size], [0, 2*marker_size], [0, 2*marker_size]], dtype=np.float32)}, 
                   '99':{'src_pts': base_src + np.array([[2*marker_size, 2*marker_size], [2*marker_size, 2*marker_size], [2*marker_size, 2*marker_size], [2*marker_size, 2*marker_size]], dtype=np.float32)}}
    for l in marker_dict.values():
        print(l['src_pts'])
    # Initialize variables for motion detection
    THRESHOLD = 99
    old_frame = None
    homography = None

    for marker_id in marker_dict.keys():
        marker_image = cv2.aruco.generateImageMarker(dictionary, int(marker_id), marker_size)
        #cv2.imwrite("marker_image_" + str(marker_id) + ".png", marker_image)
        marker_dict[str(marker_id)]['marker_image'] = marker_image
        marker_dict[str(marker_id)]['marker_image'] = marker_size
    
    cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return 

        reAnimate = True
        if old_frame is not None:
            # Calculate temporal difference matrix with threshold
            dIm = np.float32(frame) - np.float32(old_frame)
            dIm[dIm <= THRESHOLD] = 0

            if np.all(dIm == 0): # If there is no motion
                reAnimate = False

        old_frame = frame
        
        # Undistort the frame
        h,  w = frame.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # Detect markers on undistored frame
        corners, ids, _ = detector.detectMarkers(dst)

        # if there has been no motion then we don't need to reanimate the 3D model
        if reAnimate == False and homography is not None and ids is not None and len(ids) > 0:
            #aruco.drawDetectedMarkers(dst, corners)
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(newCameraMatrix, homography)  
                    # project cube or model
                    dst = render(dst, obj, projection, marker_size)
                except:
                    pass

        # Animates the 3D model on the AR markers
        elif ids is not None and len(ids) >= 2:
            #aruco.drawDetectedMarkers(dst, corners)
            src_pts = np.array([])
            dst_pts = np.array([])
            c = 0
            for id in ids:
                if c == 0:
                    src_pts = marker_dict[str(id[0])]['src_pts']
                    dst_pts = np.array(corners[0][0], dtype=np.float32)
                else:
                    
                    src_pts = np.vstack((src_pts, marker_dict[str(id[0])]['src_pts']))
                    dst_pts = np.vstack((dst_pts, corners[c][0]))
                c+=1
            
 
            #homography = DLT(src_pts, dst_pts)
            #homography *= -1
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)   
           
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(newCameraMatrix, homography)  
                    # project cube or model
                    dst = render(dst, obj, projection, marker_size)
                except:
                    pass

        # show result
        cv2.imshow('frame', dst)
        #cv2.waitKey(-1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

def DLT(pts_src, pts_dst):
    # Calculate A matrix
    for i in range(4):
        pt_src = pts_src[i]
        pt_dst = pts_dst[i]

        x, y, z = pt_src[0], pt_src[1], 1
        x_t, y_t, z_t = pt_dst[0], pt_dst[1], 1
        if i == 0:
            A = np.array([
                [0, 0, 0, -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z],
                [z_t*x, z_t*y, z_t*z, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z]
            ])
        else:
            A = np.concatenate((A, np.array([
                [0, 0, 0, -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z],
                [z_t*x, z_t*y, z_t*z, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z]
            ])), axis=0)

    # Perform SVD
    _, _, V = np.linalg.svd(A)
    h = V[-1]
    H = h.reshape((3, 3))

    return H

def render(img, obj, projection, marker_size, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.5

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + 1.5*marker_size, p[1] + 1.5*marker_size, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (100, 100, 100))
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