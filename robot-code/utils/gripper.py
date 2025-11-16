import numpy as np
import ev3_dc as ev3
from rich import print
import time
class Gripper:
    def __init__(self, port=ev3.PORT_B):
        self.port = port
        self.open = False

    def move_by(self, degrees):
        move_task = (self.motor.move_by(degrees=degrees))
        move_task.start(thread=True)

    def startup(self, ev3_obj):    
        self.motor = ev3.Motor(self.port, ev3_obj=ev3_obj, protocol="USB")
        print("[green]Connected to gripper")

    def open_gripper(self):
        if self.open:
            return False
        else:
            self.motor.start_move_by(-60, speed=5, brake=False)
            self.open = True
            return True
    
    def close_gripper(self):
        if not self.open:
            return False
        else:
            self.motor.start_move_by(60, speed=5, brake=False)
            self.open = False
            return True

