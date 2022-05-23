'''
Data recording :
This program enables synchronized data storage in the following format.
RGB-X-Y-Z-ODOM
RGB - Color map from D435i
X, Y, Z - X, Y, Z maps from D435i
ODOM - RTAB-Map odometry output
'''
import numpy as np
import pyrealsense2 as rs
import cv2
import time

import Camera
# Need to insert required packages to interface with RTAB-Map

def save_data(frame_id, save_path, rgb, x, y, z, odom):
    camera_data = np.stack(rgb, x, y, z, axis = 0)
    np.save(file = save_path + 'imgs_' + frame_id + '.npy', arr = camera_data)
    np.save(file = save_path + 'odom_' + frame_id + '.npy', arr = odom)


# camera data stream
camera = Camera()
width = 1280
height = 720
camera.start_camera(width = width, height = height, framerate = 30)
align = rs.align(rs.stream.color)
pc = rs.pointcloud()
try:
    while True:
        # start_time = time.time()
        frames = camera.wait_for_frames()
        aligned_frame = align.process(frames)
        aligned_color_frame = aligned_frame.get_color_frame()
        aligned_depth_frame = aligned_frame.get_depth_frame()
        pointsContainer = pc.calculate(aligned_depth_frame)
        points = np.asarray(pointsContainer.get_vertices())
        points = points.view(np.float32).reshape(points.shape + (-1,))
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        x = x.reshape((height,width))
        y = y.reshape((height,width))
        z = z.reshape((height,width))
        # --- Data from RTAB Map ---
        # print("--- %s Hz ---" % (1/(time.time() - start_time)))
finally:
    camera.stop()