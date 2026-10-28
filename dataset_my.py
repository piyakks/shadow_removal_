import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, load_val_img, load_mask, load_val_mask, Augment_RGB_torch
import torch.nn.functional as F
import random
from torchvision import transforms
from PIL import Image
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'train_C'
        input_dir = 'train_A'
        shadow_dir = 'train_B'
        self.img_options = img_options
        self.transform = transforms.Compose([transforms.Resize((self.img_options,self.img_options)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.gray_transform=transforms.Compose([transforms.Resize((self.img_options,self.img_options)),
                                             transforms.ToTensor(),
                                             ])
        self.gt = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        self.image = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        self.shadow_gt = sorted(os.listdir(os.path.join(rgb_dir, shadow_dir)))
        
        self.image_filename = [os.path.join(rgb_dir, input_dir, x) for x in self.image ]
        self.shadow_gt_filenames = [os.path.join(rgb_dir, shadow_dir, x) for x in self.shadow_gt ]
        self.mask_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in self.gt ]

        

        self.tar_size = len(self.gt)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        
        img = Image.open(self.image_filename[index]).convert('RGB')
        
        shadow_img = Image.open(self.shadow_gt_filenames[index]).convert('L')
        
        gt_img = Image.open(self.mask_filenames[index]).convert('RGB')
        

        clean_img = self.transform(img)
        
        mask = self.gray_transform(gt_img)
       
        
        
        shadow = self.gray_transform(shadow_img)
        

        return clean_img,shadow,mask,self.image_filename[index]
##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir,size,target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        self.size=size
        gt_dir = 'test_C'
        input_dir = 'test_A'
        shadow_dir = 'test_B'
        
        self.transform = transforms.Compose([transforms.Resize((self.size,self.size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gray_transform=transforms.Compose([transforms.Resize((self.size,self.size)),
                                             transforms.ToTensor(),
                                             ])
        self.gt = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        self.image = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        self.shadow_gt = sorted(os.listdir(os.path.join(rgb_dir, shadow_dir)))
        
        self.image_filename = [os.path.join(rgb_dir, input_dir, x) for x in self.image]
        self.shadow_gt_filenames = [os.path.join(rgb_dir, shadow_dir, x) for x in self.shadow_gt ]
        self.mask_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in self.gt ]
        
        self.tar_size = len(self.gt)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        

        img = Image.open(self.image_filename[index]).convert('RGB')
        shadow_img = Image.open(self.shadow_gt_filenames[index]).convert('L')
        gt_img = Image.open(self.mask_filenames[index]).convert('RGB')
        
        clean_img = self.transform(img)
        mask = self.gray_transform(gt_img)
        
        shadow = self.gray_transform(shadow_img)
        
        
        return clean_img,shadow,mask,self.image_filename[index]
