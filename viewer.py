import tkinter
from tkinter import filedialog as tkFileDialog
import time
import threading
import queue as Queue
import cv2
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import signal
import sys
import jsonpickle
import message # used for unpickling json str to message obj
from rich import print

from subscriber import Subscriber
from drawables import Trajectory, ScannerData, Points, Particles, WorldGrid, MapView
import argparse

parser = argparse.ArgumentParser(description='Check whether to run the viewer with a local source')
parser.add_argument('--local', required=False, help='Run with --local for local server (SLAM jupyter notebook)', action='store_true')

class GuiPart:
    def __init__(self, root, queue, endCommand):
        self.queue = queue
        self.root = root

        self.height_padding = 8
        self.width_padding = 8

        self.window_extents = (1200, 1000)
        # The extents of the world and sensor canvas.
        self.world_canvas_extents = (400, int((self.window_extents[1]- 1)/2))
        self.sensor_canvas_extents = (400, int((self.window_extents[1]- 1)/2))
        self.camera_canvas_extents = (848, int((self.window_extents[1]- 1)/2))
        self.log_canvas_extents = (400, int((self.window_extents[1]- 1)/2))
        self.map_canvas_extents = (400, int((self.window_extents[1]- 1)/2))

        # camera image size
        self.camera_img_size = (424, 240)

        self.camera_img_ratio = self.camera_img_size[0]/self.camera_img_size[1]

        # The world extents of the scene in meters
        self.world_extents = (8.0, 8.0) # world extents, in meters
        self.scanner_range = 2.0 # range of camera, in meters

        root.geometry(str(self.window_extents[0]) + "x" + str(self.window_extents[1]))
        root.title('Robot Viewer')

        # Setup GUI stuff.
        frame1 = tkinter.Frame(root)
        frame1.pack(fill=tkinter.BOTH, expand=True)
        self.world_canvas = tkinter.Canvas(frame1, height=self.world_canvas_extents[1],bg="white")
        self.world_canvas.pack(side=tkinter.LEFT, anchor=tkinter.NW, fill=tkinter.BOTH, expand=True, pady=2, padx=2)
        self.sensor_canvas = tkinter.Canvas(frame1, height=self.sensor_canvas_extents[1],bg="white")
        self.sensor_canvas.pack(side=tkinter.LEFT, anchor=tkinter.N, fill=tkinter.BOTH, expand=True, pady=2, padx=2)
        self.map_canvas = tkinter.Canvas(frame1, height=self.map_canvas_extents[1],bg="white")
        self.map_canvas.pack(side=tkinter.RIGHT, anchor=tkinter.NE, fill=tkinter.BOTH, expand=True, pady=2, padx=2)

        self.frame2 = tkinter.Frame(root)
        self.frame2.pack(fill=tkinter.BOTH, expand=True)
        self.scale = tkinter.Scale(self.frame2, orient=tkinter.HORIZONTAL, command=self.slider_moved)
        self.scale.pack(fill=tkinter.X)
        # self.info = tkinter.Label(self.frame2)
        # self.info.pack()
        # load = tkinter.Button(self.frame2,text="Load (additional) logfile",command=self.add_file)
        # load.pack(side=tkinter.LEFT)
        # reload_all = tkinter.Button(self.frame2,text="Reload all",command=self.load_data)
        # reload_all.pack(side=tkinter.RIGHT)
        # done_btn = tkinter.Button(self.frame2, text='Done', command=endCommand)
        # done_btn.pack()

        frame3 = tkinter.Frame(root)
        frame3.pack(fill=tkinter.BOTH, expand=True)
        self.camera_canvas = tkinter.Canvas(frame3,width=self.camera_canvas_extents[0], height=self.camera_canvas_extents[1],bg="gray")
        self.camera_canvas.pack(side=tkinter.LEFT, anchor=tkinter.NW, fill=tkinter.BOTH, expand=True, pady=2, padx=2)

        self.log_canvas = tkinter.Canvas(frame3, width=self.log_canvas_extents[0], height=self.log_canvas_extents[1],bg="gray")
        self.log_canvas.pack(side=tkinter.RIGHT, anchor=tkinter.NE, fill=tkinter.BOTH, expand=True, pady=2, padx=2)

        # The list of objects to draw.
        self.draw_objects = []
        self.map_view = None

        # Ask for file.
        self.all_file_names = []
        # self.add_file()
        root.update()
        root.update_idletasks()

        self.width, self.height = self.root.winfo_width(), self.root.winfo_height()
        self.compute_window_spacing()
        self.resize()

        root.bind("<Configure>", self.resize_event)

        self.load_new_data()

    
    def compute_window_spacing(self):
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()

        frame_width = self.world_canvas.winfo_width() + self.sensor_canvas.winfo_width()
        frame_height = self.world_canvas.winfo_height() + self.camera_canvas.winfo_height()

        # print(root_w - frame_width, root_h - frame_height)

        self.width_padding = root_w - frame_width
        self.height_padding = root_h - frame_height
        

    def resize_event(self, event):
        if(event.widget == self.root and (self.width != event.width or self.height != event.height)
            and (event.width != 0 or event.height != 0)):

            self.width, self.height = event.width, event.height
            self.resize()

            self.root.update_idletasks()


    
    def resize(self):

        frame_width = self.width - self.width_padding
        frame_height = self.height - self.height_padding

        # frame_width = self.world_canvas.winfo_width() + self.sensor_canvas.winfo_width()
        # frame_height = self.world_canvas.winfo_height() + self.camera_canvas.winfo_height()
        
        # print("frame_width", frame_width)
        # print("max_window_width", max_window_width)

        # print("frame_height", frame_height)
        # print("max_window_height", max_window_height)

        # now we calculate new canvas sizes
        max_canvas_width = int((frame_width)/2)
        max_canvas_height = int((frame_height)/2)
        
        self.world_canvas_extents = (max_canvas_width, max_canvas_height)
        self.sensor_canvas_extents = (max_canvas_width, max_canvas_height)

        self.world_canvas.config(width=self.world_canvas_extents[0], height=self.world_canvas_extents[1])
        self.sensor_canvas.config(width=self.sensor_canvas_extents[0], height=self.sensor_canvas_extents[1])

        camera_canvas_width = int(frame_width * (2/3))
        camera_canvas_height = frame_height - max_canvas_height

        log_canvas_width = frame_width - camera_canvas_width
        log_canvas_height = camera_canvas_height

        camera_canvas_ratio = camera_canvas_width/camera_canvas_height

        if self.camera_img_ratio > camera_canvas_ratio:
            self.camera_canvas_extents = (int(camera_canvas_width), int(camera_canvas_width/self.camera_img_ratio))

        else:
            self.camera_canvas_extents = (int(camera_canvas_height * self.camera_img_ratio), int(camera_canvas_height))

            camera_canvas_width = int(camera_canvas_height * self.camera_img_ratio)
            log_canvas_width = frame_width - camera_canvas_width


        self.camera_canvas.config(width=camera_canvas_width, height=camera_canvas_height)
        self.log_canvas.config(width=log_canvas_width, height=log_canvas_height)

    def slider_moved(self, index):
        """Callback for moving the scale slider."""
        i = int(index)
        # Call all draw objects.
        # for d in draw_objects:
        #     d.draw(i)

    def add_file(self):
        filename = tkFileDialog.askopenfilename(filetypes = [("all files", ".*"), ("txt files", ".txt")])
        if filename:
            # If the file is in the list already, remove it (so it will be appended
            # at the end).
            if filename in self.all_file_names:
                self.all_file_names.remove(filename)
            self.all_file_names.append(filename)
            self.load_data()

    def to_sensor_canvas(self, r, alpha):
        """Transforms a point from sensor coordinates (meters) to sensor canvas coord system."""
        # convert (r, alpha) to (x, y)
        rel_x = np.cos(alpha) * r
        rel_y = np.sin(alpha) * r

        # divide canvas_extents by 2 because we want to have the camera in the middle of the canvas
        scale = (1/self.scanner_range) * (np.array(self.sensor_canvas_extents)/2)

        # x and y swapped on purpose
        x = self.sensor_canvas_extents[0] / 2.0 - rel_y * scale[0]
        x = x.astype(int)
        y = self.sensor_canvas_extents[1] / 2.0 - 1 - rel_x * scale[1]
        y = y.astype(int)

        points = np.column_stack((x, y))
        return points

    def to_world_canvas(self, world_positions, single_pos=False):
        """Transforms a point from world coord system to world canvas coord system."""
        world_positions_np = np.array(world_positions)

        if single_pos:
            if len(world_positions_np) < 2:
                return np.array([])
            world_x = world_positions_np[0]
            world_y = world_positions_np[1]
        else: # len(world_positions_np.shape) == 2
            if len(world_positions_np.shape) < 2:
                return np.empty((0, 2))
            world_x = world_positions_np[:, 0]
            world_y = world_positions_np[:, 1]

        scale = (1/np.array(self.world_extents)) * np.array(self.world_canvas_extents)
        x = np.array(world_x) * scale[0] + self.world_canvas_extents[0]/2
        x = x.astype(int)
        y = self.world_canvas_extents[1]/2 - np.array(world_y) * scale[1]
        y = y.astype(int)
        # return list(points)
        if len(world_positions_np.shape) == 1:
            return np.array([x, y])
        else:
            return np.column_stack((x, y))

    def load_new_data(self, msg=None):
        draw_objects = []

        # draw axes
        draw_objects.append(ScannerData(self.sensor_canvas,
            self.sensor_canvas_extents, self.scanner_range))
        
        draw_objects.append(WorldGrid(self.world_canvas, self.world_canvas_extents, self.world_extents))

        # Initialize map view if not already done
        if self.map_view is None:
            self.map_view = MapView(self.map_canvas, self.map_canvas_extents)
            draw_objects.append(self.map_view)

        if msg is not None:
            # Update map data if available
            if hasattr(msg, 'grid_map') and len(msg.grid_map) > 0:
                self.map_view.update_map(
                    np.array(msg.grid_map), 
                    msg.grid_origin, 
                    msg.grid_resolution,
                    msg.current_path
                )

            # landmark measurements
            landmark_ids = np.array(msg.landmark_ids)
            landmark_rel_positions = self.to_sensor_canvas(msg.landmark_rs, msg.landmark_alphas)
            landmark_positions = self.to_world_canvas(msg.landmark_positions)

            # estimated landmarks positions
            landmark_estimated_ids = np.array(msg.landmark_estimated_ids)
            landmark_estimated_stdevs = np.array(msg.landmark_estimated_stdevs)
            landmark_estimated_positions = self.to_world_canvas(msg.landmark_estimated_positions)

            # estimated robot position:
            robot_position = self.to_world_canvas(msg.robot_position, single_pos=True)
            robot_stdev = np.array(msg.robot_stdev)
            robot_theta = msg.robot_theta
            robot_pose = np.append(robot_position, robot_theta)

            #! what is the error in theta?

            # print("landmark_ids", landmark_ids.shape)
            # print("landmark_rel_positions", landmark_rel_positions.shape)
            # print("landmark_positions", landmark_positions.shape)

            # print("landmark_estimated_ids", landmark_estimated_ids.shape)
            # print("landmark_estimated_stdevs", landmark_estimated_stdevs.shape)
            # print("landmark_estimated_positions", landmark_estimated_positions.shape)

            # print("robot_position", robot_position.shape)
            # print("robot_stdev", robot_stdev.shape)

            if len(landmark_rel_positions) > 0 and len(landmark_rel_positions[0]) > 0:
                draw_objects.append(Points(list([landmark_rel_positions]), self.sensor_canvas, "#262339", ids=np.expand_dims(landmark_ids, axis=0)))

            #! we add a dimension to each array. This way we *could* save all the draw objects and iterate over them...
            if robot_position is not None:
                # black points
                # draw_objects.append(Points(robot_world_points, self.world_canvas, "#262339"))
                
                # position with direction, use robot_pose instead of robot_position
                # red points with ellipse
                draw_objects.append(Trajectory(np.expand_dims(robot_pose, axis=0),
                    self.world_canvas,
                    self.world_extents,
                    self.world_canvas_extents,
                    standard_deviations=np.expand_dims(robot_stdev, axis=0),
                    cursor_color="blue", background_color="lightblue",
                    position_stddev_color = "#8080ff", theta_stddev_color="#c0c0ff"))

            # measured positions
            if len(landmark_positions) > 0:

                # only show ids for landmarks that are not yet estimated
                only_new_landmark_ids = []
                for id in landmark_ids:
                    if id in landmark_estimated_ids:
                        only_new_landmark_ids.append(None)
                    else:
                        only_new_landmark_ids.append(id)
                only_new_landmark_ids = np.array(only_new_landmark_ids)

                # black points
                draw_objects.append(Points(np.expand_dims(landmark_positions, axis=0), self.world_canvas, "#262339", ids=np.expand_dims(only_new_landmark_ids, axis=0)))

            # estimated positions
            if len(landmark_estimated_positions) > 0:

                # landmark world ellipses and red points
                scale = self.world_canvas_extents[0] / self.world_extents[0]
                draw_objects.append(Points(np.expand_dims(landmark_estimated_positions, axis=0),
                                    self.world_canvas, "#cc0000",
                                    ids=np.expand_dims(landmark_estimated_ids, axis=0),
                                    ellipses=np.expand_dims(landmark_estimated_stdevs, axis=0),
                                    ellipse_factor=scale))

        # Start new canvas and do all background drawing.
        self.world_canvas.delete(tkinter.ALL)
        self.sensor_canvas.delete(tkinter.ALL)
        self.map_canvas.delete(tkinter.ALL)

        # draw axes
        for d in draw_objects:
            d.background_draw()

        # Call all draw objects.
        for d in draw_objects:
            d.draw(0)


    def processIncoming(self):
        """Handle all messages currently in the queue, if any."""
        while self.queue.qsize():
            try:
                # we only want the latest image. delete the rest.
                # we are using a LIFO queue (last in, first out)
                msg_str, img = self.queue.get() # remove and return item from queue
                with self.queue.mutex:
                    self.queue.queue.clear()

                msg = jsonpickle.decode(msg_str)
                date_time = datetime.fromtimestamp(msg.timestamp)
                date_time_str = date_time.strftime("%m/%d/%Y, %H:%M:%S.%f")

                now = datetime.now()
                delay = now - date_time
                delay_milliseconds = int(delay.total_seconds() * 1000)


                cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.camera_img_size = img.shape[:2]
                cv2image = cv2.resize(cv2image, self.camera_canvas_extents, interpolation = cv2.INTER_NEAREST) # INTER_NEAREST is fastest
                img = Image.fromarray(cv2image)
                
                # Convert image to PhotoImage
                self.imgtk = ImageTk.PhotoImage(image = img)
                # self.camera_canvas.delete("IMG")
                self.camera_canvas.create_image(0, 0, image=self.imgtk, anchor=tkinter.NW, tags="IMG")
                
                text = ""
                text += "delay: " + str(delay_milliseconds) + "ms" + "\n"
                text += date_time_str + "\n"
                text += "img size: " + str(self.camera_img_size) + "\n"
                text += "id: " + str(msg.id) + "\n"
                text += "start: " + str(msg.start) + "\n"
                text += "landmark ids: " + str(msg.landmark_ids) + "\n"
                text += "robot pos: (" + str(np.round(msg.robot_position[0], 2)) + ", " + str(np.round(msg.robot_position[1], 2)) + ")\n"
                text += msg.text

                self.log_canvas.delete("TEXT")
                self.log_canvas.create_text(10,10,fill="black", anchor=tkinter.NW,
                        text=text, font=("Purisa", 12), tags="TEXT")

                if msg:
                    self.load_new_data(msg)

            except Queue.Empty:
                # waiting for item to be added to queue
                pass


class ThreadedClient:
    """
    Launch the main part of the GUI and the worker thread. periodicCall and
    endApplication could reside in the GUI part, but putting them here
    means that you have all the thread controls in a single place.
    """
    def __init__(self, root, local):
        """
        Start the GUI and the asynchronous threads. We are in the main
        (original) thread of the application, which will later be used by
        the GUI as well. We spawn a new thread for the worker (I/O).
        """
        self.root = root

        root.protocol("WM_DELETE_WINDOW", self.endApplication)

        # Create the queue
        self.queue = Queue.LifoQueue()

        # Set up the GUI part
        self.gui = GuiPart(root, self.queue, self.endApplication)
        
        if local:
            self.subscriber = Subscriber(port='tcp://localhost:5555') 
        else:
            self.subscriber = Subscriber(port='tcp://100.64.0.9:5555')

        # Set up the thread to do asynchronous I/O
        # More threads can also be created and used, if necessary
        self.running = 1
        # self.thread1 = threading.Thread(target=self.workerThread1)

        # Start the periodic call in the GUI to check if the queue contains
        # anything
        self.periodicCall()

        print("starting thread...")
        self.thread1 = threading.Thread(target=self.workerThread1)
        self.thread1.start()


    def periodicCall(self):
        """
        Check every 200 ms if there is something new in the queue.
        """
        self.gui.processIncoming()
        if not self.running:
            # clean up and exit
            self.subscriber.close()
            # self.root.quit()

            print("exiting...")
            sys.exit()

        self.root.after(20, self.periodicCall)

    def workerThread1(self):
        """
        This is where we handle the asynchronous I/O. For example, it may be
        a 'select(  )'. One important thing to remember is that the thread has
        to yield control pretty regularly, by select or otherwise.
        """
        while self.running:
            # Asynchronous stuff here
            msg_str, img = self.subscriber.run()
            if msg_str is not None:
                self.queue.put((msg_str, img))

    def endApplication(self):
        self.running = 0

if __name__ == '__main__':
    args = parser.parse_args()
    client = None
    root = tkinter.Tk()
    
    # root.tk.call('tk', 'scaling', 2.0) # doesn't do anything

    client = ThreadedClient(root, args.local)

    # handle ctrl + c
    def sigint_handler(sig, frame):
        print("called end application...")
        client.endApplication()

    signal.signal(signal.SIGINT, sigint_handler)

    root.mainloop()
