
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

import scipy.io as sio
from utils.loader import get_validation_data
import utils
import cv2
# from model import UNet
from mymodel_cat import mymodel
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='/mnt/d/shadow/ISTD_Dataset/test/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/ShadowFormer_istd/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=10, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--heira_mode',type=str,default='sam2_l_adapter')
parser.add_argument('--pretrain',default=True)
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir,size=256)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration = mymodel(mode=args.heira_mode,pretrained=args.pretrain)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

img_multiple_of = 8 * args.win_size

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    rmse_val_rgb = []
    psnr_val_s = []
    ssim_val_s = []
    psnr_val_ns = []
    ssim_val_ns = []
    rmse_val_s = []
    rmse_val_ns = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        img = data_test[0].cuda()
        rgb_shadow = data_test[1].cuda()
        rgb_gt = data_test[2].cuda()
        filenames = data_test[3]
        
        # Pad the input if not_multiple_of win_size * 8
        mask=rgb_shadow

        
        rgb_restored = model_restoration(img,rgb_shadow)
        rgb_restored = torch.sigmoid(rgb_restored).cpu().numpy().squeeze().transpose((1, 2, 0))
        rgb_gt= torch.clamp(rgb_gt, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        if args.cal_metrics:
            bm = torch.where(mask == 0, torch.zeros_like(mask), torch.ones_like(mask))  #binarize mask
            bm = np.expand_dims(bm.cpu().numpy().squeeze(), axis=2)

            # calculate SSIM in gray space
            gray_restored = cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2GRAY)
            gray_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
            ssim_val_rgb.append(ssim_loss(gray_restored, gray_gt, channel_axis=None, data_range=1.0))
            ssim_val_ns.append(ssim_loss(gray_restored * (1 - bm.squeeze()), gray_gt * (1 - bm.squeeze()), channel_axis=None, data_range=1.0))
            ssim_val_s.append(ssim_loss(gray_restored * bm.squeeze(), gray_gt * bm.squeeze(), channel_axis=None, data_range=1.0))

            psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
            psnr_val_ns.append(psnr_loss(rgb_restored * (1 - bm), rgb_gt * (1 - bm)))
            psnr_val_s.append(psnr_loss(rgb_restored * bm, rgb_gt * bm))

            # calculate the RMSE in LAB space
            rmse_temp = np.abs(cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2LAB)).mean() * 3
            rmse_val_rgb.append(rmse_temp)
            rmse_temp_s = np.abs(cv2.cvtColor(rgb_restored * bm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * bm, cv2.COLOR_RGB2LAB)).sum() / bm.sum()
            rmse_temp_ns = np.abs(cv2.cvtColor(rgb_restored * (1-bm), cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * (1-bm),
                                                                                                   cv2.COLOR_RGB2LAB)).sum() / (1-bm).sum()
            rmse_val_s.append(rmse_temp_s)
            rmse_val_ns.append(rmse_temp_ns)

        # cv2.imwrite(os.path.join('./results_gray/', filenames[0][39:]), (x * 255.0).astype(np.uint8))

        utils.save_img(rgb_restored*255.0, os.path.join(args.result_dir, filenames[0][39:]))

if args.cal_metrics:
    psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
    psnr_val_s = sum(psnr_val_s)/len(test_dataset)
    ssim_val_s = sum(ssim_val_s)/len(test_dataset)
    psnr_val_ns = sum(psnr_val_ns)/len(test_dataset)
    ssim_val_ns = sum(ssim_val_ns)/len(test_dataset)
    rmse_val_rgb = sum(rmse_val_rgb) / len(test_dataset)
    rmse_val_s = sum(rmse_val_s) / len(test_dataset)
    rmse_val_ns = sum(rmse_val_ns) / len(test_dataset)
    print("PSNR: %f, SSIM: %f, RMSE: %f " %(psnr_val_rgb, ssim_val_rgb, rmse_val_rgb))
    print("SPSNR: %f, SSSIM: %f, SRMSE: %f " %(psnr_val_s, ssim_val_s, rmse_val_s))
    print("NSPSNR: %f, NSSSIM: %f, NSRMSE: %f " %(psnr_val_ns, ssim_val_ns, rmse_val_ns))

