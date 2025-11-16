from __future__ import annotations
import cv2
import numpy as np
from rich import print
from utils.opencv_utils import putBText
from scipy.spatial.transform import Rotation
from scipy import optimize
from enum import Enum
from utils.utils import boundary


class Vision:
    def __init__(self, camera_matrix, dist_coeffs, cam_config) -> None:

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.cam_config = cam_config
        

        # Camera to robot transform
        self.x_offset_camera = 0.03
        self.y_offset_camera = 0.00
        self.z_offset_camera = 0.28
        self.x_angle_camera = -124 # degrees # 90,-180,90 ## 35,0,160
        self.y_angle_camera = 0
        self.z_angle_camera = -90  # -90 degrees
        
        # Robot to world transform
        self.x_offset_world = 0.4         # 0.4,-0.25, 0, -30,0,0
        self.y_offset_world = -0.25
        self.z_offset_world = 0
        self.x_angle_world = -30  # degrees
        self.y_angle_world = 0
        self.z_angle_world = 0
        
        # Create transformation matrices
        self.offset_camera_robot = np.array([self.x_offset_camera, self.y_offset_camera, self.z_offset_camera])
        self.rotation_camera_robot = np.array([
            np.radians(self.x_angle_camera),
            np.radians(self.y_angle_camera),
            np.radians(self.z_angle_camera)
        ])
        
        self.offset_robot_world = np.array([self.x_offset_world, self.y_offset_world, self.z_offset_world])
        self.rotation_robot_world = np.array([
            np.radians(self.x_angle_world),
            np.radians(self.y_angle_world),
            np.radians(self.z_angle_world)
        ])
        
        # Initialize transformation matrices
        self.T_camera_robot = self.vectors_to_transformation_matrix(
            self.rotation_camera_robot, self.offset_camera_robot)
        self.T_robot_world = self.vectors_to_transformation_matrix(
            self.rotation_robot_world, self.offset_robot_world)
        
    def rotation_matrix(self, x_rotation, y_rotation, z_rotation):
        alpha = z_rotation
        beta = y_rotation
        gamma = x_rotation

        # https://en.wikipedia.org/wiki/Rotation_matrix
        # rotates around z, then y, then x

        return np.array([
            [np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
            [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
            [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]
        ])
        
    def vectors_to_transformation_matrix(self, rotation, translation):
        ### Your code here ###
        R = self.rotation_matrix(*rotation)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation.flatten()
        ###
        return T

    def transformation_matrix_to_vectors(self, T):
        ### Your code here ###
        R = T[:3, :3]
        translation = tuple(T[:3, 3])
        beta = -np.arcsin(R[2, 0])  # Y-rotation

        if np.abs(R[2, 0]) < 1 - 1e-6:  # Not at a singularity
            alpha = np.arctan2(R[1, 0], R[0, 0])  # Z-rotation
            gamma = np.arctan2(R[2, 1], R[2, 2])  # X-rotation
        else:  # Gimbal lock (singularity)
            alpha = np.arctan2(-R[0, 1], R[1, 1])  # Z-rotation
            gamma = 0  # X-rotation is set to 0
        ###
        return np.array([gamma, beta, alpha]), translation

    def detections(self, img: np.ndarray, draw_img: np.ndarray, robot_pose: tuple, marker_size: float = 0.05) -> tuple:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        marker_corners, ids, rejected = aruco_detector.detectMarkers(img)
        
        #print(marker_corners, ids)
        ids_list, landmark_rs, landmark_alphas, landmark_positions = [], [], [], []
        
        if ids is not None:
            # Define marker points
            marker_points = np.array([
                [-marker_size/2, marker_size/2, 0],
                [marker_size/2, marker_size/2, 0],
                [marker_size/2, -marker_size/2, 0],
                [-marker_size/2, -marker_size/2, 0]
            ])
            
            draw_img = cv2.aruco.drawDetectedMarkers(draw_img, marker_corners, ids)
            
            # Process each marker
            for i, corners in enumerate(marker_corners):
                success, rvec, tvec = cv2.solvePnP(
                    marker_points, corners[0], self.camera_matrix, self.dist_coeffs)
                
                if success:
                    rvec , tvec = np.squeeze(rvec), np.squeeze(tvec)
                    
                    # Transform to robot frame
                    T_marker_camera = self.vectors_to_transformation_matrix(rvec, tvec)
                    T_marker_robot = self.T_camera_robot @ T_marker_camera
                    
                    # Get current robot pose transformation
                    x, y, theta, _ = robot_pose
                    current_robot_rotation = np.array([0, 0, theta])
                    current_robot_translation = np.array([x, y, 0])
                    T_current_robot = self.vectors_to_transformation_matrix(
                        current_robot_rotation, current_robot_translation)
                    
                    # Calculate distances and angles in robot frame first
                    robot_rvec, robot_tvec = self.transformation_matrix_to_vectors(T_marker_robot)
                    x, y = robot_tvec[0], robot_tvec[1]
                    r = np.sqrt(x*x + y*y)
                    alpha = np.arctan2(y, x)
                    
                    # Transform to world frame
                    T_robot_world_inv = np.linalg.inv(self.T_robot_world)
                    T_marker_world = T_current_robot @ T_marker_robot
                    
                    # Get world position
                    world_rvec, world_tvec = self.transformation_matrix_to_vectors(T_marker_world)
                    
                    ids_list.append(ids[i][0])
                    landmark_rs.append(r)
                    landmark_alphas.append(alpha)
                    landmark_positions.append((world_tvec[0], world_tvec[1]))
                    label = f"ID:{ids[i][0]} r:{r:.2f}m α:{np.degrees(alpha):.1f}°"
                else:
                    print(f"Pose estimation failed for marker {i}")
                    
        return (ids_list, landmark_rs, landmark_alphas, landmark_positions)