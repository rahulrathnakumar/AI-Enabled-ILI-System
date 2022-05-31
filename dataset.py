import os
import random
import unittest

from cv2 import resize
from sympy import root
import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib
# matplotlib.use('tkAgg')
from matplotlib import pyplot as plt

from transforms_utils import RandomChoice

torch.manual_seed(17)

class DefectDataset(Dataset):
    '''
    An Extensible Multi-Modality Dataset Class
    Notes:
    Offline inference:
     -- Check filename format for extracting train val sample data.
     -- Currently, we organize the data needed in separate folders (Eg: Image, Depth, Normal, Curvature).
      
    Streaming inference:
     -- Dataloading not impl yet.
    '''
    def __init__(self, root_dir, num_classes, input_modalities = ['images', 'depth', 'normal', 'curvature'], image_set='train', num_training = None,  transforms=None, labels_avail = False):
        self.n_class = num_classes
        self.root_dir = root_dir
        self.image_set = image_set
        self.transforms = transforms
        self.input_modalities = input_modalities # Needs to have same name as path (assert this)
        self.labels_avail = labels_avail
        if self.labels_avail:
            # self.target_filenames = []
            self.input_modalities.append('labels')
        self.data_filenames = dict((key, []) for key in self.input_modalities)
        if num_training and image_set == 'train':
            self.reference_filename = os.path.join(root_dir, '{image_set}_{num_training}samples.txt'.format(image_set = image_set, num_training = num_training))
            img_list = np.random.choice(self.read_image_list(self.reference_filename), num_training, replace= False)
        else:
            self.reference_filename = os.path.join(root_dir, '{:s}.txt'.format(image_set))
            img_list = self.read_image_list(self.reference_filename)
        for img_name in img_list:
            for modality in self.input_modalities:
                if os.path.isfile(os.path.join(root_dir, modality + '/{:s}'.format(img_name))):
                    self.data_filenames[modality].append(os.path.join(root_dir, modality + '/{:s}'.format(img_name)))

    def read_image_list(self, filename):
        list_file = open(filename, 'r')
        img_list = []
        while True:
            next_line = list_file.readline()
            if not next_line:
                break
            img_list.append(next_line.rstrip())
        return img_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = dict((key, []) for key in self.input_modalities)
        for key in data:
            if key == 'normal' or key == 'images':
                data[key] = Image.open(self.data_filenames[key][index]).convert('RGB')
            else:
                data[key] = Image.open(self.data_filenames[key][index]).convert('L')

        if self.transforms is not None:
            transform = RandomChoice(self.transforms)
            for key in data: data[key] = transform([data[key]])

        # Data transform : resize, convert to tensor and normalize
        resize_data = transforms.Resize((224,224))
        totensor = transforms.ToTensor()
        normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize_L = transforms.Normalize(mean=[0.485], std=[0.229])
        resize_target = transforms.Resize((224,224), interpolation=Image.NEAREST)

        for key in data:
            if key != 'labels':
                data[key] = resize_data.__call__(data[key])
                fmt = data[key].mode
                data[key] = totensor(data[key])
                if fmt == 'RGB':
                    data[key] = normalize_rgb(data[key])
                if fmt == 'L':
                    data[key] = normalize_L(data[key])
            else:   
                data[key] = resize_target.__call__(data[key])
                data[key] = np.asarray(data[key])
                data[key] = torch.from_numpy(data[key]).long()
                h,w = data[key].size()
                label = torch.zeros(self.n_class, h, w)
                for c in range(self.n_class):
                    label[c][data[key] == c] = 1
                data[key] = label
        return data

if __name__ == "__main__":
    dataset = DefectDataset(root_dir = '/home/rrathnak/Documents/Work/Task-2/Datasets/asuDataset', num_classes = 3, num_training=8, labels_avail=True)
    data = dataset.__getitem__(1)
