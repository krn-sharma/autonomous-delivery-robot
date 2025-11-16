"""publisher.py -- send PiCamera image stream, compressing images.

publisher
"""

import sys

import socket
import time
from datetime import datetime
import traceback
import cv2
import imagezmq
import jsonpickle
import subprocess
from message import Message


class Publisher():
    def __init__(self) -> None: 
        # use either of the formats below to specifiy address of display computer
        # sender = imagezmq.ImageSender(connect_to='tcp://100.76.2.68:5555')

        # PUB/SUB:
        self.sender = None

        self.jpeg_quality = 50  # 0 to 100, higher is better quality, 95 is cv2 default

    def __enter__(self):
        self.sender = imagezmq.ImageSender(connect_to='tcp://*:5555', REQ_REP=False)

    def publish_img(self, msg, image):

        # image = self.print_on_image(image)
        # processing of image before sending would go here.
        # for example, rotation, ROI selection, conversion to grayscale, etc.
        ret_code, jpg_buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

        self.sender.send_jpg(msg, jpg_buffer)

    def publish_str(self, msg):
        self.sender.zmq_socket.send_string(msg)

    def close(self):
        self.sender.close()  # close the ZMQ socket and context

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

if __name__ == '__main__':
    publisher = Publisher()

    rpi_name = socket.gethostname()  # send RPi hostname with each image

    subprocess.call("v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_auto_priority=0 -c exposure_absolute=100", shell=True)

    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 424) # more FOV than with 640
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    vc.set(cv2.CAP_PROP_FPS, 30)

    time.sleep(1.0)  # allow camera sensor to warm up

    count = 0
    try:
        while True:  # send images as stream until Ctrl-C
            rval, image = vc.read()
            msg = Message()
            msg.id = count
            # msg = rpi_name + " " + str(count)
            msg_str = jsonpickle.encode(msg)

            publisher.publish_img(msg_str, image)
            count += 1
    except (KeyboardInterrupt, SystemExit):
        pass  # Ctrl-C was pressed to end program
    except Exception as ex:
        print('Python error with no Exception handler:')
        print('Traceback error:', ex)
        traceback.print_exc()
    finally:
        vc.release()  # stop the camera thread
        publisher.close()
        sys.exit()
