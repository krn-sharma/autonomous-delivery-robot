import cv2
import time
import numpy as np
import sys
import jsonpickle
import pickle
from message import Message
from timeit import default_timer as timer
from numba import typed

from utils.robot_controller import RobotController
from utils.exploration import Exploration
from utils.thinking import Thinking
from utils.follow_path import PathFollower
from utils.grid_map import GridMap
from utils.gripper import Gripper

from utils.a_star import a_star
from utils.a_star import simplify_path

from publisher import Publisher
from utils.keypress_listener import KeypressListener
from rich import print
from utils.utils import load_config
from types import SimpleNamespace
import os


from enum import Enum
class TaskPart(Enum):
    """
    A helper Enum for the mode we are in.
    """
    Manual = 0
    Exploration = 1
    ToStartLine = 2
    Race = 3
    Load = 4
    Planning = 5  # New mode for block movement
    Thinking = 6
    TransportBlockToGate = 7  # New mode for transporting blocks to gates
    PathFollower = 8
    Parking = 9

class Main():
    def __init__(self) -> None:
        """
        
        """

        # load config
        self.config = load_config("config.yaml")

        # Grid map configuration from config file
        self.grid_resolution = self.config.main.grid_resolution
        self.grid_width = self.config.main.grid_width
        self.grid_height = self.config.main.grid_height
        
        # instantiate methods
        self.robot = RobotController(self.config)
        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        
        self.rotation_start = 0

        # set default values
        self.DT = self.config.robot.delta_t # delta time in seconds

        self.speed = 0
        self.turn = 0
        self.new_speed = 0
        self.new_turn = 0

        # Add to Main.__init__
        self.SAFE_DISTANCE = 0.4
        self.APPROACH_ANGLE = np.pi/3
        self.MIN_MARKER_SPACING = 0.6
        
        # Add main instance to config for exploration
        self.config.main_instance = self
        self.exploration = Exploration(self.config)
        self.thinking = Thinking()

        
        self.last_viz_time = 0
        self.viz_interval = 60.0
        self.map_save_dir = "maps"

        self.pickup = True
        self.current_color = None
        self.delivered_blocks = []

        # Planning mode variables
        self.current_block = None
        self.current_gate = None
        self.is_blue_turn = True  # Start with blue blocks
        self.block_state = 'FIND_BLOCK'  # States: FIND_BLOCK, APPROACH_BLOCK, PUSH_TO_GATE
        
        self.manualMode = False
        self.is_running = True

        self.map = None

        self.mode = TaskPart.Manual

        self.run_loop()

    def run_loop(self):
        """
        this loop wraps the methods that use the __enter__ and __close__ functions:
            self.keypress_listener, self.publisher, self.robot
        
        then it calls run()
        """
        print("starting...")

        # control vehicle movement and visualize it
        with self.keypress_listener, self.publisher, self.robot:
            print("starting EKF SLAM...")

            print("READY!")
            print("[green]MODE: Manual")

            count = 0

            while self.is_running:
                time0 = timer()
                self.run(count, time0)

                elapsed_time = timer() - time0
                if elapsed_time <= self.DT:
                    dt = self.DT - elapsed_time
                    time.sleep(dt) # moves while sleeping
                else:
                    print(f"[red]Warning! dt = {elapsed_time}")

                count += 1

            print("*** END PROGRAM ***")

    def run(self, count, time0):
        """
        Were we get the key press, and set the mode accordingly.
        We can use the robot recorder to playback a recording.
        """


        if not self.robot.recorder.playback:
            # read webcam and get distance from aruco markers
            _, raw_img, cam_fps, img_created = self.robot.camera.read() # BGR color

            speed = self.speed
            turn = self.turn
        else:
            cam_fps = 0
            raw_img, speed, turn = next(self.robot.recorder.get_step)

        if raw_img is None:
            print("[red]image is None!")
            return

        if self.mode == TaskPart.Race:
            draw_img = raw_img
            data, self.red_blocks, self.blue_blocks = self.robot.run_ekf_slam(raw_img, fastmode = True)
        else:
            draw_img = raw_img.copy()
            data, self.red_blocks, self.blue_blocks  = self.robot.run_ekf_slam(raw_img, draw_img)

        self.parse_keypress()

        if self.mode == TaskPart.Manual:
            self.robot.move(self.speed, self.turn)

        if self.mode == TaskPart.Exploration:
            self.robot.gripper.close_gripper()
            
            if self.rotation_start == 0:
                self.rotation_start = time.time()

            elapsed_time = time.time() - self.rotation_start

            if elapsed_time < 30:
                step_duration = 2  # Wartezeit zwischen den Drehungen (in Sekunden)
                turn_step = 10     # Drehwinkel pro Schritt (kann angepasst werden)

                if int(elapsed_time) % step_duration == 0:  # Alle `step_duration` Sekunden drehen
                    print(f"[green]Rotating step {elapsed_time:.1f} seconds")
                    self.speed, self.turn = 0, turn_step
                    self.robot.move(self.speed, self.turn)
                else:
                    self.robot.move(0, 0)  # Warten (Stillstand)
            else:
                if self.delivered_blocks == self.config.main.blocks_to_deliver:
                    self.mode = TaskPart.Manual

                self.speed, self.turn, end_reached = self.exploration.tramp(data)
                
                if end_reached or len(self.delivered_blocks) == self.config.main.blocks_to_deliver:
                    print('[green]Exploration complete - Switching to Manual mode')
                    self.mode = TaskPart.Manual
                
                undelivered_red_blocks = self.exploration.red_blocks.copy()
                undelivered_blue_blocks = self.exploration.blue_blocks.copy()

                for delivered_block in self.delivered_blocks:
                    if delivered_block in undelivered_red_blocks:
                        del undelivered_red_blocks[undelivered_red_blocks]
                    elif delivered_block - 1 in undelivered_red_blocks:
                        del undelivered_red_blocks[delivered_block - 1]
                    elif delivered_block in undelivered_blue_blocks:
                        del undelivered_blue_blocks[delivered_block]
                    elif delivered_block - 1 in undelivered_blue_blocks:
                        del undelivered_blue_blocks[delivered_block - 1]
                print(
                    "gate", self.exploration.gate_positions,
                    "delivered_blocks",self.delivered_blocks,
                    "current_color",self.current_color,
                    "undelivered_red_blocks",undelivered_red_blocks,
                    "undelivered_blue_blocks",undelivered_blue_blocks
                )

                if (self.exploration.gate_positions['red'] is not None and len(self.exploration.gate_positions['red']) > 1 and len(undelivered_red_blocks) > 0 and (self.current_color is None or self.current_color == "red")):
                    
                    print('[green]Found red gate and blocks - Switching to Thinking mode')
                    self.current_color = "red"
                    self.mode = TaskPart.Thinking



                elif (self.exploration.gate_positions['blue'] is not None and len(self.exploration.gate_positions['blue']) > 1 and len(undelivered_blue_blocks) > 0 and (self.current_color is None or self.current_color == "blue")):

                    print('[green]Found blue gate and blocks - Switching to Thinking mode')
                    self.current_color = "blue"              
                    self.mode = TaskPart.Thinking

                else:
                    # Continue exploration if no matching pairs found
                    self.robot.move(self.speed, self.turn)
                    
                    if end_reached:
                        print('[green]Exploration complete - Switching to Manual mode')
                        self.mode = TaskPart.Manual

        if self.mode == TaskPart.Thinking:
            print(self.current_color, self.red_blocks)
            undelivered_red_blocks = self.exploration.red_blocks.copy()
            undelivered_blue_blocks = self.exploration.blue_blocks.copy()
            
            for delivered_block in self.delivered_blocks:
                if delivered_block in undelivered_red_blocks:
                    del undelivered_red_blocks[delivered_block]
                elif delivered_block - 1 in undelivered_red_blocks:
                    del undelivered_red_blocks[delivered_block - 1]
                elif delivered_block in undelivered_blue_blocks:
                    del undelivered_blue_blocks[delivered_block]
                elif delivered_block - 1 in undelivered_blue_blocks:
                    del undelivered_blue_blocks[delivered_block - 1]
                    
            red_block_ids = []
            blue_block_ids = []
            for block in undelivered_red_blocks.values():
                red_block_ids.append(block['marker2']['id'])
            for block in undelivered_blue_blocks.values():
                blue_block_ids.append(block['marker2']['id'])
            
            if self.pickup == True: # wants to pickup a block
                self.robot.gripper.open_gripper()

                if self.current_color == "red":
                    self.current_block = self.thinking.find_nearest_block(self.current_color, data, red_block_ids)

                elif self.current_color == "blue":
                    self.current_block = self.thinking.find_nearest_block(self.current_color, data, blue_block_ids)

                print(f'[green]Thinking path to {self.current_color} Block with ID- {self.current_block}')
                block_location = self.thinking.get_block_position(self.current_block,data)  #world coord
                print(block_location)
                routes = self.thinking.a_star(block_location,data) #grid coord
                self.waypoints = self.thinking.simplify_path(routes)   #grid coord
                self.path = [self.thinking.grid_to_world(x, y) for x, y in self.waypoints] # world coord
                print(self.path, block_location)

                if len(self.path) == 0:
                    print('[red]Path is empty')
                    self.mode = TaskPart.Manual
                print('[green]Path Found')
                self.mode = TaskPart.PathFollower

            else: # wants to deliver a block
                self.robot.gripper.close_gripper()
                time.sleep(1)
                print('[green]Thinking path to Gate')
                self.exploration.detect_gate(data)
                gate_location = self.exploration.gate_positions[self.current_color]
                self.drop_location,self.gate_orientation = self.thinking.compute_drop_location(gate_location,(data.robot_position[0],data.robot_position[1]))
                routes = self.thinking.a_star(self.drop_location,data)
                self.waypoints = self.thinking.simplify_path(routes)
                self.path = [self.thinking.grid_to_world(x, y) for x, y in self.waypoints]
                print("Path to Gate:",routes, self.path)
                print('[green]Path Found')
                print("color", self.current_color)
                print("Gate location", gate_location)
                print("drop location", self.drop_location)
                print("Gate location", self.gate_orientation)
                print("robot location", (data.robot_position[0],data.robot_position[1]))
                self.mode = TaskPart.PathFollower


        if self.mode == TaskPart.PathFollower:
            path_follower = PathFollower(self.path,self.config)
            #self.thinking.display_map(path = self.waypoints)
            if self.pickup:
                self.speed, self.turn, arrived = path_follower.follow_path(data,0.12)  # wants to stop 12 cm away from the block 
            
            else:
                self.speed, self.turn, arrived = path_follower.follow_path(data,0,self.gate_orientation)

            self.robot.move(self.speed,self.turn)
            if arrived:
                print('[green]Destination Arrived')

                if not self.pickup:
                    self.drive_for_duration(15, 0, 4)
                    self.robot.gripper.open_gripper()
                    self.drive_for_duration(-15, 0, 4)
                    self.delivered_blocks.append(self.current_block)
                    self.current_color = 'blue' if self.current_color == 'red' else 'red'
                    self.robot.move(0,0)
                    self.pickup = not self.pickup
                    self.mode = TaskPart.Exploration
                    #self.mode = TaskPart.Thinking
                else:
                    self.pickup = not self.pickup
                    print(self.pickup)
                    self.mode = TaskPart.Thinking
                

        # create a message for the viewer
        msg = Message(
            id = count,
            timestamp = time0,
            start = True,

            landmark_ids = data.landmark_ids,
            landmark_rs = data.landmark_rs,
            landmark_alphas = data.landmark_alphas,
            landmark_positions = data.landmark_positions,

            landmark_estimated_ids = data.landmark_estimated_ids,
            landmark_estimated_positions = data.landmark_estimated_positions,
            landmark_estimated_stdevs = data.landmark_estimated_stdevs,

            robot_position = data.robot_position,
            robot_theta = data.robot_theta,
            robot_stdev = data.robot_stdev,

            text = f"cam fps: {cam_fps}"
        )

        msg_str = jsonpickle.encode(msg)

        # send message to the viewer
        self.publisher.publish_img(msg_str, draw_img)


    def save_state(self, data):
        self.robot.slam.dump()


    def drive_for_duration(self, speed, turn, duration, control_interval=0.1):
        """
        Command the robot to move with a given speed and turn for a specific duration.
        control_interval: time between successive commands
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            self.robot.move(speed, turn)
            time.sleep(control_interval)
        # Stop the robot after moving
        self.robot.move(0, 0)


    def load_and_localize(self):
        with open("SLAM_practice.pkl", 'rb') as file:
            loaded_data = pickle.load(file)
        ids = loaded_data['ids']
        index_to_ids = loaded_data['index_to_ids']
        n_ids = loaded_data['n_ids']
        mu = loaded_data['mu']
        sigma = loaded_data['sigma']
        # to avoid map destruction upon map loading and repositioning
        sigma[0, 0] = 100000000
        sigma[1, 1] = 100000000
        sigma[2, 2] = 100000000

        return (ids, index_to_ids, n_ids, mu, np.copy(sigma))

    def parse_keypress(self):
        char = self.keypress_listener.get_keypress()

        turn_step = 40
        speed_step = 5

        if char == "a":
            if self.turn >= 0:
                self.new_turn = self.turn + turn_step
            else:
                self.new_turn = 0
            self.new_turn = min(self.new_turn, 200)
        elif char == "d":
            if self.turn <= 0:
                self.new_turn = self.turn - turn_step
            else:
                self.new_turn = 0
            self.new_turn = max(self.new_turn, -200)
        elif char == "w":
            if self.speed >= 0:
                self.new_speed = self.speed + speed_step
            else:
                self.new_speed = 0
            self.new_speed = min(self.new_speed, 100)
        elif char == "s":
            if self.speed <= 0:
                self.new_speed = self.speed - speed_step
            else:
                self.new_speed = 0
            self.new_speed = max(self.new_speed, -100)
        elif char == "c":
            self.new_speed = 0
            self.new_turn = 0
        elif char == "q":
            self.new_speed = 0
            self.new_turn = 0
            self.is_running = False
        elif char == "m":
            self.new_speed = 0
            self.new_turn = 0
            self.mode = TaskPart.Manual
            print("[green]MODE: Manual")
        elif char == "r":
            self.mode = TaskPart.Race
            print("[green]MODE: Race")
        elif char == "l":
            self.mode = TaskPart.Load
            print("[green]MODE: Load map")
        elif char == "p":
            self.mode = TaskPart.ToStartLine
            print("[green]MODE: To start line")
        elif char == "e":
            self.mode = TaskPart.Exploration
            print("[green]MODE: Exploration")
        elif char == "t":
            self.mode = TaskPart.Thinking
            print("[green]MODE: Thinking")
        
        elif char == "o":
            self.robot.gripper.open_gripper()

        elif char == "c":
            self.robot.gripper.close_gripper()

        

        if self.speed != self.new_speed or self.turn != self.new_turn:
            self.speed = self.new_speed
            self.turn = self.new_turn


if __name__ == '__main__':

    main = Main()

