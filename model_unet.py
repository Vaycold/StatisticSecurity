import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from adv_patch_gen.utils.dataset import YOLODataset
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet_cbam_early import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM, PSNR
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
import time
from glob import glob
from tqdm import tqdm
from datetime import datetime
import pytz
import warnings
import torch.nn as nn
warnings.filterwarnings('ignore')

korea_time = 'Asia/Seoul'
tz = pytz.timezone(korea_time)

def read_image(path) -> torch.Tensor:
    """
    Read an input image to be used as a patch

    Arguments:
        path: Path to the image to be read.
    """
    patch_img = Image.open(path).convert("RGB")
    adv_patch_cpu = T.ToTensor()(patch_img)
    return adv_patch_cpu


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

        
        
        
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'conv'):
            nn.init.normal_(m.conv.weight, 0.0, 0.02)
            if hasattr(m.conv, 'bias') and m.conv.bias is not None:
                nn.init.constant_(m.conv.bias, 0)
                print('bias 적용 안됨!!!!!!')
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'bn') and m.bn is not None:
            nn.init.normal_(m.bn.weight, 1.0, 0.02)
            nn.init.constant_(m.bn.bias, 0)
            
            
def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    cur_time = datetime.now(tz)
    run_name = cur_time.strftime("%Y%m%d_%H%M")
    
    abspath = os.path.abspath(os.getcwd())+'/'
    
    print('저장되는 경로 : ',abspath,os.path.join(args.checkpoint_path, f'{run_name}'))
    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.cuda()
    model.apply(weights_init)
    
    modulename =['ssmtcb','sspcab','cbam',]
    
    for mn in modulename :
        for name, module in model.named_modules() :

            if mn in name :
                print('##########################################################')
                print()      
                print(f'##################### {mn} 적용#########################')
                print()      
                print('##########################################################')
                break

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_psnr = PSNR()

    dataset = YOLODataset(image_dir = args.data_path,
                          label_dir = args.label_path, 
                          max_labels = 48,
                          model_in_sz = [640, 640],
                          shuffle = True)

    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    patch_dirs = glob(os.path.join(args.patch_paths, '*.png'))
    
    savelog_per_step_by_batchsize = ((len(dataloader) // args.bs)*args.epochs) // 650
    n_iter = 0
    args.visualize = True
    for epoch in range(args.epochs):
        print("Epoch: "+str(epoch))
        for i_batch, (sample_batched, label_batched) in enumerate(tqdm(dataloader)):
            
            patch = random.choice(patch_dirs)
            
            adv_patch_cpu = read_image(path=patch)
            
            patch_transformer = PatchTransformer([0.25, 0.4], [0.5, 0.8], 0.1, [-0.25, 0.25], [-0.25, 0.25], torch.device("cuda")).cuda()
            patch_applier = PatchApplier(1).cuda()                

            gray_batch = sample_batched.cuda()
            label_batch = label_batched.cuda()
            adv_patch = adv_patch_cpu.cuda()
            adv_batch_t = patch_transformer(adv_patch, label_batch,
                                            model_in_sz=[640, 640],
                                            use_mul_add_gau="all",
                                            do_transforms=True,
                                            do_rotate=True,
                                            rand_loc=True)
            p_img_batch = patch_applier(gray_batch, adv_batch_t)
            aug_gray_batch = F.interpolate(p_img_batch, (640, 640))


            gray_rec = model(aug_gray_batch)                

            l2_loss = loss_l2(gray_rec,gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)
            psnr_loss = loss_psnr(gray_rec, gray_batch)
            loss = l2_loss + ssim_loss + psnr_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if args.visualize and n_iter % savelog_per_step_by_batchsize == 0:
                visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                visualizer.plot_loss(psnr_loss, n_iter, loss_name='psnr_loss')
                
            if args.visualize and n_iter % savelog_per_step_by_batchsize == 0:
                #t_mask = out_mask_sm[:, 1:, :, :]
                visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
                visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')


            n_iter +=1
            del patch, adv_patch_cpu, patch_transformer, patch_applier
        scheduler.step()
        
        if (epoch+1) % 100 == 0 or epoch == 0 :
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f'{run_name}_{epoch+1}.pckl'))



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    
    
    parser.add_argument('--data_path', default='/Data1/security/EXP3/Dataset/images')
    parser.add_argument('--label_path', default='/Data1/security/EXP3/Dataset/labels')
    parser.add_argument('--patch_paths', default='/Data1/hm/DRAEM_TRAIN_PATCH')
    
    #parser.add_argument('--model_name', default='/home/user/Workplace/last/Dataset/patch')

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)

