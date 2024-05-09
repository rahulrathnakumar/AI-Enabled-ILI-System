'''
Data recording :
This program enables synchronized inference using PyTorch.
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
ODOM - RTAB-Map odometry output:
Odom format: 3x3 matrix : [gyro 1x3, accel 1x3, stepper_odom 1x3 ]
stepper_odom : [0 0 angle]
=====
'''
import numpy as np
import pyrealsense2 as rs
import cv2
import time
import urllib.request
import argparse
import os


import matplotlib.pyplot as plt
# ROS


from Camera import *
from data_utils import *

