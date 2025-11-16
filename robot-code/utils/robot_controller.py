from __future__ import annotations
from types import SimpleNamespace
import ev3_dc as ev3
import numpy as np
from rich import print

from utils.camera import Camera
from utils.vision import Vision
from utils.EKFSLAM import EKFSLAM
from utils.recorder import Recorder
from utils.gripper import Gripper
from utils.robot_dummy import DummyVehicle

from timeit import default_timer as timer


class RobotController:
    def __init__(self, config) -> None:

        self.config = config
        self.dt = config.robot.delta_t

        self.__ev3_obj__ = None
        self.vehicle = None

        self.camera = None
        self.vision = None

        self.gripper = None

        self.recorder = Recorder(self.dt)

        self.slam = EKFSLAM(
            config.robot.wheel_radius,
            config.ekf_slam.robot_width,
            MOTOR_STD=config.ekf_slam.motor_std,
            DIST_STD=config.ekf_slam.dist_std,
            ANGLE_STD=config.ekf_slam.angle_std
        )

        self.red_block_pairs = [(i, i + 1) for i in range(421, 450, 2)]  
        self.blue_block_pairs = [(i, i + 1) for i in range(521, 550, 2)] 

        self.old_l, self.old_r = 0, 0
        self.past_ids=[]

        self.detected_ids = set()
        self.landmark_sightings = {}
        self.min_sightings = 1


    def __enter__(self) -> RobotController:

        self.camera = Camera(self.config.camera.exposure_time,
                             self.config.camera.gain)
        self.vision = Vision(self.camera.CAMERA_MATRIX, self.camera.DIST_COEFFS,
                             self.config.camera)

        try:
            self.__ev3_obj__ = ev3.EV3(protocol=ev3.USB, sync_mode="STD")
        except Exception as e:
            print("error:", e)

        if self.__ev3_obj__:
            self.vehicle = ev3.TwoWheelVehicle(
                self.config.robot.wheel_radius, # radius wheel
                self.config.robot.width, # middle-to-middle tread measured
                speed = 10,
                ev3_obj=self.__ev3_obj__
            )

            self.gripper = Gripper()
            self.gripper.startup(ev3_obj= self.__ev3_obj__)

            print("[green]***CONNECTED TO REAL VEHICLE***[/green]")
        else:
            print("[red]***USING Dummy VEHICLE***[/red]")
            self.vehicle = DummyVehicle()


        

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.recorder.is_recording:
            self.recorder.save_recording()

            if self.vehicle:
                self.vehicle.stop(brake=False)
                self.vehicle.__exit__(exc_type, exc_val, exc_tb)

            if self.__ev3_obj__:
                self.__ev3_obj__.__exit__(exc_type, exc_val, exc_tb)

        if self.camera:
                self.camera.close()

    def wheel_rotation_from_distance(self, d):
        """
        @brief: Calculates the rotation of a robot wheel from the distance that wheel has traveled and the wheel diameter
        @param d: Distance traveled
        @return: Angle by which the wheel has turned while traveling distance d
        """
        wheel_circumference = 2*self.slam.WHEEL_RADIUS*np.pi
        revolutions = d / wheel_circumference

        return 2*np.pi * revolutions
    
    def move(self, speed, turn, img=None, dt=None):
        x, y, theta, _ = self.slam.get_robot_pose()
        if not dt:
            dt = self.dt

        v_max = 1
        w_max = np.pi
        v = np.clip(v_max * speed / 100, -1 * v_max, v_max)
        w = np.clip(w_max * turn / 100, -1 * w_max, w_max)
        if speed == 0 and turn == 0:
            self.vehicle.stop(brake=False)
            return
        if w == 0:
            vl = v
            vr = v
        else:
            vl = w*(v/w - self.slam.WIDTH/2)
            vr = w*(v/w + self.slam.WIDTH/2)

        vr = int(np.ceil(100 * vr))
        vl = int(np.ceil(100 * vl))
        ops = b''.join((
            ev3.opOutput_Step_Speed,
            ev3.LCX(0),  # LAYER
            ev3.LCX(self.vehicle.port_left),  # NOS
            ev3.LCX(vl),
            ev3.LCX(50),  # STEP1
            ev3.LCX(50),  # STEP2
            ev3.LCX(50),  # STEP3
            ev3.LCX(0),  # BRAKE
            
            ev3.opOutput_Step_Speed,
            ev3.LCX(0),  # LAYER
            ev3.LCX(self.vehicle.port_right),  # NOS
            ev3.LCX(vr),
            ev3.LCX(50),  # STEP1
            ev3.LCX(50),  # STEP2
            ev3.LCX(50),  # STEP3
            ev3.LCX(0),  # BRAKE
            
            ev3.opOutput_Start,
            ev3.LCX(0),  # LAYER
            ev3.LCX(self.vehicle.port_left + self.vehicle.port_right),  # NOS
        ))

        self.vehicle.send_direct_cmd(
                self.vehicle._ops_pos() + ops,
                global_mem=8
            )
        


        self.recorder.save_step(img, speed, turn, x, y, theta, self.old_l, self.old_r)

    def move_old(self, speed, turn, img=None):
        self.vehicle.move(speed, turn)
        self.recorder.save_step(img, speed, turn)

    def get_motor_movement(self) -> tuple:
        """Get the current motor positions in radians"""
        if self.__ev3_obj__:
            l = self.vehicle.motor_pos.left
            r = self.vehicle.motor_pos.right
            
            l = self.slam.WHEEL_RADIUS * np.deg2rad(l)
            r = self.slam.WHEEL_RADIUS * np.deg2rad(r)
        else:
            l = self.vehicle.l
            r = self.vehicle.r
            
        return (l, r)

    def run_ekf_slam(self, img, draw_img=None, fastmode=False):
        l, r = self.get_motor_movement()

        movements = l - self.old_l, r - self.old_r
        self.old_l, self.old_r = l, r
        if movements[0] != 0 or movements[1] != 0:
            self.slam.predict(*movements)

        ids, landmark_rs, landmark_alphas, landmark_positions = self.vision.detections(img, draw_img, self.slam.get_robot_pose())

        current_ids = set(ids)
        for id in list(self.landmark_sightings.keys()):
            if id not in current_ids:
                del self.landmark_sightings[id]

        # Process each detected landmark
        for i, id in enumerate(ids):
            self.landmark_sightings[id] = self.landmark_sightings.get(id, 0) + 1
            
            if id not in self.slam.get_landmark_ids():

                if self.landmark_sightings[id] >= self.min_sightings:
                    self.slam.add_landmark(landmark_positions[i], id)

                    print(f"Landmark with id {id} added after {self.min_sightings} sightings")
            else:
                # correct each detected landmark that is already added
                self.slam.correction((landmark_rs[i], landmark_alphas[i]), id)

        def extract_blocks(landmark_ids, block_pairs):
            detected_blocks = []
            
            for pair in block_pairs:
                if pair[0] in landmark_ids and pair[1] in landmark_ids:
                    detected_blocks.append(pair[1])  # Store the larger ID (closer to the block)
            
            return detected_blocks


        robot_x, robot_y, robot_theta, robot_stdev = self.slam.get_robot_pose()
        landmark_estimated_positions, landmark_estimated_stdevs  = self.slam.get_landmark_poses()
        landmark_estimated_ids = self.slam.get_landmark_ids()
                
        red_blocks = extract_blocks(landmark_estimated_ids, self.red_block_pairs)
        blue_blocks = extract_blocks(landmark_estimated_ids, self.blue_block_pairs)
        
        data = SimpleNamespace()
        data.landmark_ids = ids
        data.landmark_rs = landmark_rs
        data.landmark_alphas = landmark_alphas
        data.landmark_positions = landmark_positions
        data.landmark_estimated_ids = landmark_estimated_ids
        data.landmark_estimated_positions = landmark_estimated_positions
        data.landmark_estimated_stdevs = landmark_estimated_stdevs
        
        data.grid_map = []
        data.grid_origin = []
        data.grid_resolution = 0.0
        data.current_path = []


        data.robot_position = np.array([robot_x, robot_y])
        data.robot_theta = robot_theta
        data.robot_stdev = robot_stdev

        return data, red_blocks, blue_blocks
