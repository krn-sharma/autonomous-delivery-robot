"""
1. run publisher.py on raspberry pi
2. run subscriber.py on PC
"""

import sys

import time
import traceback
import numpy as np
import cv2
from collections import defaultdict
from imutils.video import FPS
import imagezmq

class Subscriber():
    def __init__(self, port='tcp://100.64.0.9:5555') -> None:

        # pub/sub:
        #self.image_hub = imagezmq.ImageHub(open_port='tcp://100.64.0.1:5555', REQ_REP=False)
        self.image_hub = imagezmq.ImageHub(open_port=port, REQ_REP=False)

        self.image_count = 0
        self.first_image = True

        self.fps = None

    def run(self):
        try:
            msg, jpg_buffer = self.image_hub.recv_jpg()

        except Exception as e:
            print("caught exception", e)
            return None, None
        # print("msg", msg)

        if self.first_image:
            print("Connected to raspberry pi!")
            self.fps = FPS().start()  # start FPS timer after first image is received
            self.first_image = False
        image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)

        self.fps.update()
        self.image_count += 1  # global count of all images received

        return msg, image

    def close(self):
        print()
        print('Total Number of Images received: {:,g}'.format(self.image_count))

        if self.fps is not None:
            self.fps.stop()
            print('Elasped time: {:,.2f} seconds'.format(self.fps.elapsed()))
            print('Approximate FPS: {:.2f}'.format(self.fps.fps()))
        self.image_hub.close()  # closes ZMQ socket and context

if __name__ == '__main__':
    subscriber = Subscriber()

    try:
        while True:
            msg, img = subscriber.run()
            cv2.imshow("rpi", img)  # display images 1 window per sent_from
            cv2.waitKey(1)
    except (KeyboardInterrupt, SystemExit):
        pass  # Ctrl-C was pressed to end program; FPS stats computed below
    except Exception as ex:
        print('Python error with no Exception handler:')
        print('Traceback error:', ex)
        traceback.print_exc()
    finally:
        subscriber.close()
