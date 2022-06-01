'''
inference.py
Functionality:
The goal of this program is to accept data from the Camera stream, load in a trained ML model, and produce predictions,
uncertainties, measurements and risk assessments. 
**** This is the OFFLINE INFERENCE CODE for hardware-software integration ****
Expected IO: 
Input: Data in the form of images
Output: Corresponding results, convert to GIF for demo.
===============================================
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
from risk_assessment import measure, ASME_B31g, NG18


use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Args
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help = 'OPTIONS: Pipeline, RoadCracks', default = 'Pipeline')
parser.add_argument("--data_fmt", help = 'OPTIONS: rgb, rgb-d, rgb-dc, rgb-dnc', default = 'rgb-dnc')
parser.add_argument("--model_path", help = 'full path to inference model weights', default= 'models/')
parser.add_argument("--root_dir", help = "Root directory for images", default = 'data/rec_0/')
parser.add_argument("--mc_samples", help = 'Number of MC Dropout samples (int) ', default = '10')
args = parser.parse_args()

dataset = args.dataset
data_fmt = args.data_fmt
model_path = args.model_path
mc_samples = args.mc_samples
root_dir = args.root_dir
p = 0.5 # Dropout ratio
batch_size = 32 # max this out!
'''
Data FMT:
RGB-X-Y-Z-Odom
This data is stored offline using record.py during inspection run.
- Data processing w/ numpy to obtain RGB-DNC from RGB-XYZ
- PyTorch dataloader 
- Proceed as usual for inference
- Plug in measurement module (improvements needed!)
- Risk assessment from measurement module (improvements needed!)

'''

# Dataset params
dataset_options = ['Pipeline, RoadCracks']
assert dataset in dataset_options, "Dataset error."
if dataset == 'Pipeline':
    num_classes = 4
else:
    num_classes = 2


# Load model 
vgg_model = VGGNet()
if data_fmt == 'rgbd':
    input_modalities = ['images', 'depth']
    net = FCNDepth(pretrained_net = vgg_model, n_class = num_classes, p = p)
elif data_fmt == 'rgbdnc':
    input_modalities = ['images', 'depth', 'normal', 'curvature']
    net = FCNDNC(pretrained_net = vgg_model, n_class = num_classes, p = p)
elif data_fmt == 'rgbdc':
    input_modalities = ['images', 'depth', 'curvature']
    net = FCNDC(pretrained_net = vgg_model, n_class = num_classes, p = p)
else:
    input_modalities = ['images']
    net = FCN(pretrained_net = vgg_model, n_class = num_classes, p = p)

net, epoch = load_ckp(model_path, net)
val_dataset = DefectDataset(root_dir = root_dir, num_classes = num_classes, input_modalities = input_modalities,
image_set = 'val')
val_dataloader = DataLoader(val_dataset, batch_size= batch_size, shuffle=False)
vgg_model = vgg_model.to(device)
net = net.to(device)
net.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
print("Validating at epoch: {:.4f}".format(epoch))
with torch.no_grad():
    net.dropout.train()
    softmax = nn.Softmax(dim = 1)
    for iter, data in enumerate(val_dataloader):
        sampled_outs = []
        outs_sm = []
        input = input.to(device)
        label = label.to(device)
        # Predicted aleatoric variance from a single pass
        aleatoric_uncertainty = net(input)[:, num_classes:, :, :]
        assert aleatoric_uncertainty.shape[1] == num_classes, "Aleatoric uncertainty shape error."
        aleatoric_uncertainty = np.exp(aleatoric_uncertainty.detach().clone().cpu().numpy())
        # Sampled epistemic uncertainty
        for i in range(mc_samples):
            sampled_outs.append(net(input))
        for out in sampled_outs:
            N, _, h, w = out.shape
            out_ = out[:, num_classes, :, : ]
            out__ = softmax(out_)
            outs_sm.append(out__.cpu().numpy())
            # compute_metrics(out_, label.detach().clone())
        
        mean_output = np.mean(np.stack(outs_sm), axis = 0) # TODO: Threshold this mean output (int?)
        epistemic_uncertainty = np.mean(np.stack(outs_sm), axis = 0) - (np.mean(np.stack(outs_sm), axis = 0))**2 # Batches x num_classes x W x H
        classwise_epistemic_uncertainty = np.mean(epistemic_uncertainty, axis = (2,3))
        classwise_aleatoric_uncertainty = np.mean(aleatoric_uncertainty, axis = (2,3))
        measurements = measure(mean_output)
