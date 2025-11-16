import numpy as np
from rich import print
from utils.pid_controller import PIDController
import matplotlib.pyplot as plt


class PathFollower:
    def __init__(self, path, config):
        self.speed = 0

        # load parameters for PID controller
        kp = 0.5  #config.pid.kp
        ki = 0.01 #config.pid.ki
        kd = 0.3  #config.pid.kd
        self.turn = 0
        self.removal_distance = 0.30 #config.pid.removal_distance  # 0.25 m

        # # start the computationof the path
        # print('Computing the path')
        self.path = path

        # # save the image, but takes some seconds to complete
        # if config.maze.save_fig:
        #     self.save_path_info(data)

        # print('Path computed, ready to run the race')

        self.pid_turn = PIDController(kp, ki, kd)

    # Calculate the direction vector from the current pose to the target coordinate
    def compute_desired_direction(self, current_pose, target_coordinate):
        dx = target_coordinate[0] - current_pose[0]
        dy = target_coordinate[1] - current_pose[1]
        desired_direction = np.arctan2(dy, dx)
        return desired_direction

    # calculate the angle between the current orientation and the desired one
    def compute_error_angle(self, desired_direction, actual_direction):

        desired_direction = (desired_direction + np.pi) % (2 * np.pi) - np.pi
        actual_direction = (actual_direction + np.pi) % (2 * np.pi) - np.pi
        # Calculate the error angle between the desired and actual directions
        error_angle = desired_direction - actual_direction
        # Normalize the error angle to be within [-pi, pi]
        error_angle = (error_angle + np.pi) % (2 * np.pi) - np.pi
        return error_angle

    # Check if the robot has reached its target
    def reached_target_coordinate(self, target, robot_pose,final_distance):
        #print("DISTANCE TO TARGET:", np.sqrt(np.sum((target-robot_pose)**2)))
        # if closer than 5 cm form the checkpoint

        if np.sqrt(np.sum((target-robot_pose)**2)) < final_distance:
            return True
        else:
            return False


    def align_to_gate(self, data, gate_orientation):
        # Get the robot's current orientation in radians
        robot_angle = data.robot_theta

        # Calculate both possible desired orientations (perpendicular to the gate)
        desired_orientation1 = gate_orientation + np.pi / 2
        desired_orientation2 = gate_orientation - np.pi / 2

        # Compute the error angles for both possibilities
        error1 = self.compute_error_angle(desired_orientation1, robot_angle)
        error2 = self.compute_error_angle(desired_orientation2, robot_angle)

        # Choose the error that requires the minimal turn
        error_angle = error1 if abs(error1) < abs(error2) else error2

        # Return the angle difference in degrees
        return np.degrees(error_angle)


    # actually compute what to do at every cycle
    def follow_path(self, data, final_distance, gate_orientation=None, dt=0.1):
        #print(self.path)
        # get current position and orientation
        robot_pose = data.robot_position
        robot_angle = data.robot_theta
        robot_angle = (robot_angle + np.pi) % (2 * np.pi) - np.pi

        if len(self.path) == 0:
            print("End reached")

            if gate_orientation:
                error = self.align_to_gate(data,gate_orientation)
                if error < 20:
                    return 0,0,True
                else:
                    return 0,error,False
            return 0, 0, True
        
        target_position = self.path[0]

        orientation_target = target_position

        desired_direction = self.compute_desired_direction(
            robot_pose, orientation_target)
        actual_direction = robot_angle

        error_angle = self.compute_error_angle(desired_direction, actual_direction)
        self.turn = self.pid_turn.update(error_angle, dt)

        self.turn = int(np.degrees(self.turn))

        dx = self.path[0][0] - robot_pose[0]
        dy = self.path[0][1] - robot_pose[1]
        distance = np.hypot(dx, dy) - final_distance


        # Kp = 40
        # max_speed = 20  # Set a maximum speed based on your robot's limits.
        # self.speed = int(max(0, min(Kp * distance, max_speed)))
        self.speed = 5 + int(10 * min(distance, 1.0))

        if self.turn > 60:
            self.speed = 0

        if self.reached_target_coordinate(self.path[0], robot_pose, 0.05):
            self.speed = 0
            print('Removed point, going to next')
            if (len(self.path) > 0):
                self.path.pop(0)

        return self.speed, self.turn, False

    # save map and computed path
    def save_path_info(self, data):
        ids = data.landmark_estimated_ids
        positions = data.landmark_estimated_positions
        robot_pose = data.robot_position
        for pos_zip in zip(positions, ids):
            col = pos_zip[1] % 3
            if col == 0:
                col2 = 'green'
            elif col == 1:
                col2 = 'red'
            elif col == 2:
                col2 = 'blue'
            if pos_zip[1] < 100:
                col2 = 'black'
            plt.scatter(*pos_zip[0], color=col2)

        for point in self.path:
            plt.scatter(*point, color="purple")

        plt.scatter(*self.path[0], c='red', label='start path')
        plt.scatter(*self.path[-1], c='blue', label='end end')
        plt.scatter(*self.path[3], c='coral', label='intermediate path')
        plt.scatter(*robot_pose, color='orange', label='robot pose')
        plt.gca().set_aspect('equal')
        plt.legend(bbox_to_anchor=(1.05,  1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig('path.png')
        print('Figure saved')