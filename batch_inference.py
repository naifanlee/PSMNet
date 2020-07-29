from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= './VO04_L.png',
                    help='load model')
parser.add_argument('--rightimg', default= './VO04_R.png',
                    help='load model')                                      
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()
        if args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()     
        
        with torch.no_grad():
            disp = model(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp


def main():

        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])    

        imgL_path = os.path.join(args.datapath, 'left')
        imgR_path = os.path.join(args.datapath, 'right')
        imgL_blur_path = os.path.join(args.datapath, 'left_blur')
        imgR_blur_path = os.path.join(args.datapath, 'right_blur')
        imgL_remap_path = os.path.join(args.datapath, 'left_remap')
        imgR_remap_path = os.path.join(args.datapath, 'right_remap')

        img_disp_path = os.path.join(args.datapath, 'disp')
        img_blur_disp_path = os.path.join(args.datapath, 'blur')
        img_remap_disp_path = os.path.join(args.datapath, 'remap')
        for subpath in [img_blur_disp_path, img_remap_disp_path]:
            os.system('rm -rf {}'.format(subpath))
            os.mkdir(subpath)


        for fn in sorted(os.listdir(imgL_blur_path)):
            imgL_fpath = os.path.join(imgL_path, fn)
            imgR_fpath = os.path.join(imgR_path, fn)
            imgL_blur_fpath = os.path.join(imgL_blur_path, fn)
            imgR_blur_fpath = os.path.join(imgR_blur_path, fn)
            imgL_remap_fpath = os.path.join(imgL_remap_path, fn)
            imgR_remap_fpath = os.path.join(imgR_remap_path, fn)
            for imgL_fpath_infer, imgR_fpath_infer in [(imgL_fpath, imgR_fpath), (imgL_blur_fpath, imgR_blur_fpath), (imgL_remap_fpath, imgR_remap_fpath)]:

            # for imgL_fpath_infer, imgR_fpath_infer in [(imgL_blur_fpath, imgR_blur_fpath), (imgL_remap_fpath, imgR_remap_fpath)]:
            #for imgL_fpath_infer, imgR_fpath_infer in [(imgL_fpath, imgR_fpath)]:
                # imgL_o = cv2.imread(imgL_fpath_infer)
                # imgR_o = cv2.imread(imgR_fpath_infer)
                # print(infer_transform(imgL_o).shape)
                imgL_o = Image.open(imgL_fpath_infer).convert('RGB')
                imgR_o = Image.open(imgR_fpath_infer).convert('RGB')
                # imgL_o = imgL_o.crop((100, 100, 356, 356))
                # imgR_o = imgR_o.crop((100, 100, 356, 356))
                # print(infer_transform(imgL_o).size)

                imgL = infer_transform(imgL_o)
                imgR = infer_transform(imgR_o) 
            

                # pad to width and hight to 16 times
                if imgL.shape[1] % 16 != 0:
                    times = imgL.shape[1]//16       
                    top_pad = (times+1)*16 -imgL.shape[1]
                else:
                    top_pad = 0
                print(imgL.shape)
                if imgL.shape[2] % 16 != 0:
                    times = imgL.shape[2]//16                       
                    right_pad = (times+1)*16-imgL.shape[2]
                else:
                    right_pad = 0    

                imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
                imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

                start_time = time.time()
                pred_disp = test(imgL,imgR)
                print('time = %.2f' %(time.time() - start_time))

                
                if top_pad !=0 or right_pad != 0:
                    img = pred_disp[top_pad:,:-right_pad]
                else:
                    img = pred_disp
                
                if imgL_fpath_infer == imgL_fpath:
                    img_disp_fpath = os.path.join(img_disp_path, fn)
                if imgL_fpath_infer == imgL_blur_fpath:
                    img_disp_fpath = os.path.join(img_blur_disp_path, fn)
                if imgL_fpath_infer == imgL_remap_fpath:
                    img_disp_fpath = os.path.join(img_remap_disp_path, fn)
                
                plt.imsave(img_disp_fpath, img, cmap='plasma')
                plt.close() 

if __name__ == '__main__':
   main()






