import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from sklearn.metrics import roc_auc_score
from utils import *
import random

import argparse


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=128, help='height of input images')
parser.add_argument('--w', type=int, default=128, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')# 2
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ff++', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

# train_folder = args.dataset_path+"/"+args.dataset_type+"/training/frames"
# test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"
# train_folder = args.dataset_path+"/"+args.dataset_type+"/train"
# test_folder = args.dataset_path+"/"+args.dataset_type+"/test"
train_folder = '/root/data/c40_pred_facedata/train'
# test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"
# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

# test_dataset = DataLoader(test_folder, transforms.Compose([
#              transforms.ToTensor(),
#              ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

train_size = len(train_dataset)  # 2486
print('train size:', train_size)
# test_size = len(test_dataset)  # 2010

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
# test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
#                              shuffle=False, num_workers=args.num_workers_test, drop_last=False)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


loss_mse = CustomLoss()
train_losses = []
# Model setting
assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
if args.method == 'pred':
    from model.Pred import *

    model = VAE_CNN_PRED()
else:
    from model.Reconstruction import *
    model = VAE_CNN_RECON()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model.cuda()

# Training

for epoch in range(args.epochs):
    labels_list = []
    model.train()
    train_loss = 0
    start = time.time()
    for j, (imgs) in enumerate(train_batch):
        imgs = Variable(imgs).cuda()  # imgs [4,15,256,256]
        if args.method == 'pred':
            outputs, mu, logvar= model.forward(imgs[:, 0:12])  # imgs[:, 0:12][4,12,256,256]
        else:
            outputs, mu, logvar= model.forward(imgs)
        optimizer.zero_grad()
        if args.method == 'pred':
            loss = loss_mse(outputs, imgs[:, 12:], mu, logvar)
        else:
            loss = loss_mse(outputs, imgs)
        # loss = loss.requires_grad_()
        # loss.backward(retain_graph=True)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if j % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, j * len(imgs), len(train_dataset), loss.item() / len(imgs)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_dataset)))
    torch.save(model, os.path.join('/root/data/MNAD-master-vae/weights', '%d_model.pth' % epoch))
    # scheduler.step()
    # if epoch % 20 == 0:
    #     torch.save(model, os.path.join('D:/pycharmprojects/MNAD-master/weights', '%d_model.pth' % (epoch)))
    # print('----------------------------------------')
    # print('Epoch:', epoch+1)
    # if args.method == 'pred':
    #     print('Loss: Prediction {:.6f}'.format(loss.item()))
    # else:
    #     print('Loss: Reconstruction {:.6f}'.format(loss.item()))
    # print('----------------------------------------')
print('Training is finished')



