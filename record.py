'''
Data recording :
This program enables synchronized data storage in the following format.
We also use an extremum seeking controller (ESC) to perform online 
optimization of the depth map. 
Note on ESC:
In the current implementation, ESC does not always run, but only
when the quality metric (loss) increases a specific threshold above the 
local optimum. 
=====
DATA FMT:
RGB-X-Y-Z-ODOM
RGB - Color map from D435i
X, Y, Z - X, Y, Z maps from D435i
ODOM - RTAB-Map odometry output
=====
'''
import numpy as np
import pyrealsense2 as rs
import cv2
import time

# ROS
import rospy

import Camera
# Need to insert required packages to interface with RTAB-Map

np.seterr(all='raise')


def save_data(frame_id, save_path, rgb, x, y, z, odom):
    camera_data = np.stack(rgb, x, y, z, axis = 0)
    np.save(file = save_path + 'imgs_' + frame_id + '.npy', arr = camera_data)
    np.save(file = save_path + 'odom_' + frame_id + '.npy', arr = odom)

def get_frame(camera):
    frames = camera.wait_for_frames()
    aligned_frame = align.process(frames)
    aligned_color_frame = aligned_frame.get_color_frame()
    aligned_depth_frame = aligned_frame.get_depth_frame()
    return aligned_color_frame, aligned_depth_frame



def compute_fill_factor(imgs):
    '''
    Given K frames, compute time-averaged fill factor fitness.
    Input: imgs: KxWxH array : either RoI or entire Image
    Returns: fill_factor_cost
    '''
    # For one frame
    num_filled = np.asarray([np.count_nonzero(img) for img in imgs])
    fill_factor = np.asarray([(nf/(imgs.shape[1]*imgs.shape[2])) for nf in num_filled])
    fill_factor_cost = np.mean(np.asarray([-np.log(f) if f != 0 else 100.0 for f in fill_factor]))
    return fill_factor_cost

def ES_step(p_n,i,cES_now,amplitude):
    # ES step for each parameter
    p_next = np.zeros(nES)
    
    # Loop through each parameter
    for j in np.arange(nES):
        p_next[j] = p_n[j] + amplitude*dtES*np.cos(dtES*i*wES[j]+kES*cES_now)*(aES[j]*wES[j])**0.5
    
        # For each new ES value, check that we stay within min/max constraints
        if p_next[j] < -1.0:
            p_next[j] = -1.0
        if p_next[j] > 1.0:
            p_next[j] = 1.0
            
    # Return the next value
    return p_next

# Function that normalizes paramters
def p_normalize(p):
    p_norm = 2.0*(p-p_ave)/p_diff
    return p_norm

# Function that un-normalizes parameters
def p_un_normalize(p):
    p_un_norm = p*p_diff/2.0 + p_ave
    return p_un_norm


# camera data stream
camera = Camera()
width = 1280
height = 720
camera.start_camera(width = width, height = height, framerate = 30)

# ESC params init
eps = 1e-3
p_min, p_max = camera.query_sensor_param_bounds()
nES = len(p_min)
p_diff = p_max - p_min
p_ave = (p_max + p_min)/2.0
# pES = np.zeros([ES_steps,nES])
wES = np.linspace(1.0,1.75,nES)
dtES = 2*np.pi/(10*np.max(wES))
oscillation_size = 0.1
aES = wES*(oscillation_size)**2
kES = 0.1
decay_rate = 0.99
amplitude = 1.0

align = rs.align(rs.stream.color)
pc = rs.pointcloud()
frame_id = 0
try:
    while True:
        frame_id = frame_id + 1
        start_time = time.time()
        color_frame, depth_frame = get_frame(camera)
        depth = np.asarray(depth_frame.get_data())
        cost = compute_fill_factor(depth_frame)
        prev_cost = cost
        while cost - prev_cost > eps:
            print("Hold Camera still ... ESC working.")
            frame_id = frame_id + 1
            pES = camera.get_depth_sensor_params()
            pES_n = p_normalize(pES)
            pES_n = ES_step(pES_n,cost,amplitude)
            pES = p_un_normalize(pES_n)
            prev_cost = cost
            cost = compute_fill_factor(depth_frame)
            if cost - prev_cost <= eps:
                print("Camera param calibration complete.")
        pointsContainer = pc.calculate(depth_frame)
        points = np.asarray(pointsContainer.get_vertices())
        points = points.view(np.float32).reshape(points.shape + (-1,))
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        x = x.reshape((height,width))
        y = y.reshape((height,width))
        z = z.reshape((height,width))
        # --- Data from RTAB Map --- #
        '''
        Notes:
        Insert relevant calls to data outs from rtabmap-ros for odom data
        '''

        # Call save_data to write data into file (continuous? too much space(!!))


        # print("--- %s Hz ---" % (1/(time.time() - start_time)))
finally:
    camera.stop()