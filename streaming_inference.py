'''
inference.py
Functionality:
The goal of this program is to accept data from the Camera stream, load in a trained ML model, and produce predictions,
uncertainties, measurements and risk assessments. 
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
from dataset import *



