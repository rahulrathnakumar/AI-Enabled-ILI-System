'''
inference.py
Functionality:
The goal of this program is to accept data from the Camera stream, load in a trained ML model, and produce predictions,
uncertainties, measurements and risk assessments. 
This is the OFFLINE INFERENCE CODE for hardware-software integration.
How would this possibly work?
- Camera streaming at 30Hz.
- Capture images when stepper rotates: You would end up having a whole set of images to do detection on. From the point of view
of inference, if time is not a concern, we can repeat inference on each new frame, but this is inefficient. MC Dropout is slow. 
We need to measure the speed of MC Dropout. The uncertainty quantification is the limiting factor. 
- Once you do inference, how do you mark and count unique instances of defects? Change detection is an idea we can use here. 
For detecting changes offline, we need to have, along with each frame, the metadata of odometry. This would be an estimate of 
angular position along with its uncertainty.  
- 

'''
# Torch
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms     
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Core
import numpy as np
import cv2
import time
import sys
import os
import copy
import GPUtil
import shutil
import csv
import time 
import argparse

# Visualization packages
from visdom import Visdom
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import cm

# Modules
from network import *
from data_handler import *

