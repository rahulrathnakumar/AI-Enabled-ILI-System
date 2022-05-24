from cv2 import resize
import torch
import torchvision
import numpy as np
import os
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torchvision.transforms.functional as TF

import matplotlib
# matplotlib.use('tkAgg')
from matplotlib import pyplot as plt
import utils

torch.manual_seed(17)

class ASUDataset(Dataset):
    '''
    An Extensible Multi-Modality Dataset Class
    Notes:
    Check filename format for extracting train val sample data.
    '''
    def __init__(self, root_dir, num_classes, input_modalities = ['rgb', 'depth', 'normal', 'curvature'], image_set='train', num_training = None,  transforms=None):
        self.n_class = num_classes
        self.root_dir = root_dir
        self.image_set = image_set
        self.transforms = transforms
        self.input_modalities = input_modalities
        self.data = dict((key, []) for key in self.input_modalities)
        
        # This part of the code is for a fixed train set. Randomly sampling train sets need another way ... Need to decide on format.
        if num_training and image_set == 'train':
            self.reference_filename = os.path.join(root_dir, '{image_set}_{num_training}samples.txt'.format(image_set = image_set, num_training = num_training))
            img_list = np.random.choice(self.read_image_list(self.reference_filename), num_training, replace= False)
        else:
            self.reference_filename = os.path.join(root_dir, '{:s}.txt'.format(image_set))
            img_list = self.read_image_list(self.reference_filename)

        for img_name in img_list:
            img_filename = os.path.join(root_dir, 'images/{:s}'.format(img_name))
            target_filename = os.path.join(root_dir, 'labels/{:s}'.format(img_name))

            if os.path.isfile(img_filename) and os.path.isfile(target_filename):
                self.images.append(img_filename)
                self.targets.append(target_filename)

            

        # self.images = []
        # self.depths = []
        # self.targets = []
        # self.normals = []
        # self.curvatures = []

        # if num_training and image_set == 'train':
        #     self.reference_filename = os.path.join(root_dir, '{image_set}_{num_training}samples.txt'.format(image_set = image_set, num_training = num_training))
        #     img_list = np.random.choice(self.read_image_list(self.reference_filename), num_training, replace= False)
        # else:
        #     self.reference_filename = os.path.join(root_dir, '{:s}.txt'.format(image_set))
        #     img_list = self.read_image_list(self.reference_filename)


        # for img_name in img_list:
        #     img_filename = os.path.join(root_dir, 'images/{:s}'.format(img_name))
        #     depth_filename = os.path.join(root_dir, 'depth/{:s}'.format(img_name))
        #     target_filename = os.path.join(root_dir, 'labels/{:s}'.format(img_name))
        #     normal_filename = os.path.join(root_dir, 'normal/{:s}'.format(img_name))
        #     curvature_filename = os.path.join(root_dir, 'curvature/{:s}'.format(img_name))

        #     if os.path.isfile(img_filename) and os.path.isfile(target_filename) and os.path.isfile(depth_filename):
        #         self.images.append(img_filename)
        #         self.depths.append(depth_filename)
        #         self.targets.append(target_filename)
        #         self.normals.append(normal_filename)
        #         self.curvatures.append(curvature_filename)

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
        image = Image.open(self.images[index]).convert('RGB')
        depth = Image.open(self.depths[index]).convert('L')
        target = Image.open(self.targets[index]).convert('L')
        normal = Image.open(self.normals[index]).convert('RGB')
        curvature = Image.open(self.curvatures[index]).convert('L')
        

        if self.transforms is not None:
            transform = RandomChoice(self.transforms)
            image, depth, normal, curvature, target = transform([image, depth, normal, curvature, target])
        # Data transform : resize, convert to tensor and normalize
        resize_data = transforms.Resize((224,224))
        totensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize_normal = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize_depth = transforms.Normalize(mean=[0.485], std=[0.229])
        normalize_curvature = transforms.Normalize(mean=[0.406], std=[0.225])

        # Perform transformations on depth image [TRY TO STANDARDIZE THIS PIECE OF CODE INTO self.transforms as above for RGB]
        depth = resize_data.__call__(depth)
        depth = totensor(depth)
        depth = normalize_depth(depth)
        normal = resize_data.__call__(normal)
        normal = totensor(normal)
        normal = normalize_normal(normal)
        curvature = resize_data.__call__(curvature)
        curvature = totensor(curvature)
        curvature = normalize_curvature(curvature)
        image = resize_data.__call__(image)
        image = totensor.__call__(image)
        image = normalize.__call__(image)


        # Resize target, Convert target to numpy and one-hot encode
        resize_target = transforms.Resize((224,224), interpolation=Image.NEAREST)
        target = resize_target.__call__(target)
        target = np.asarray(target)

        target = torch.from_numpy(target).long()
        h,w = target.size()
        
        label = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            label[c][target == c] = 1
        return image, depth, normal, curvature, target, label
