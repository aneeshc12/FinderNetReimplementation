import open3d as o3d 
import numpy as np 
import copy
import torch
import math
import argparse
import os
import pandas as pd
import csv
import torch.optim as optim
import matplotlib.pyplot as plt 
from torchvision import transforms
import torch.nn as nn
import sys
import torch.nn.functional as F
import cv2

class end2endModel(nn.Module):
    def __init__(self):
        super(end2endModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        '''enc dec'''
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        ''' DYT, embedding transformer, and pixel wise dist implemented as functions '''

        ''' Delta layer and CNNs '''
        self.enricherCNN = EnricherCNN(4, ResBottleneckBlock, [2, 2, 2, 2], useBottleneck=False, outputs=256)
        self.distanceCNN = DistanceCNN()

    ''' compare both embds and extract the optimal yaw
        return rotation
    '''

    def displayDEM(self, DEM, title="img", wait=True):
        DEM_display = cv2.applyColorMap(np.array(DEM * 256, dtype=np.uint8), cv2.COLORMAP_JET)
        cv2.imshow(title, DEM_display)
        # if wait:
        #     if cv2.waitKey(0) == ord('q'):
        #         cv2.destroyAllWindows()

    def DYT(self, emb1, emb2):
        batchSize, _, h, w = emb1.size()

        ''' get conversion indices, interpolated '''
        grid = torch.zeros((batchSize, h, w, 2)).to(self.device)
        ulim = (0, np.sqrt(2.))
        vlim = (-np.pi, np.pi) 
        urange = torch.linspace(ulim[0], ulim[1], w, device=self.device)
        vrange = torch.linspace(vlim[0], vlim[1], h, device=self.device)
        vs, us = torch.meshgrid([vrange, urange], indexing='ij')
        xs = us * torch.cos(vs) 
        ys = us * torch.sin(vs)

        # print(us[-8:,-8:])
        # print()
        # print(vs[-8:,-8:])
        # print()
        # print(xs.shape, ys.shape)
        # print()
        # print(xs[:8,:8])
        # print()
        # print(ys[:8,:8])

        polarGrid = torch.stack([xs, ys], 2, out=None)
        
        for batch in range(batchSize):
            grid[batch] = polarGrid

        polarEmb1 = F.grid_sample(emb1, grid, align_corners=True)
        polarEmb2 = F.grid_sample(emb2, grid, align_corners=True)



        ''' get the optimal R value '''
        flippedEmbs = torch.flip(polarEmb1, [2])
        flippedEmbs = polarEmb1

        searchEmbs = torch.cat((polarEmb1, flippedEmbs), dim=2).to(self.device)
        
        # print("embs: ", emb1.squeeze(0)[0].cpu().detach().numpy())
        # print("sdfgs: ", emb1.squeeze(0)[0].cpu().detach().numpy())

        # self.displayDEM(emb1.squeeze(0)[0].cpu().detach().numpy(), "original DEM")
        # self.displayDEM(polarEmb1.squeeze(0)[0].cpu().detach().numpy(), "transformed DEM")

        # if cv2.waitKey(0) == ord('q'):
        #     cv2.destroyAllWindows()

        # self.displayDEM(polarEmb1.squeeze(0)[0].cpu().detach().numpy())
        # self.displayDEM(polarEmb2.squeeze(0)[0].cpu().detach().numpy())
        
        # print(polarEmb1.shape)
        # self.displayDEM(flippedEmbs.squeeze(0)[0].cpu().detach().numpy())
        
        # self.displayDEM(polarEmb2.squeeze(0)[0].cpu().detach().numpy(), wait=False)
        # self.displayDEM(searchEmbs.squeeze(0)[0].cpu().detach().numpy(), "search space layer 0")
        
        # self.displayDEM(polarEmb2.squeeze(0)[1].cpu().detach().numpy(), wait=False)
        # self.displayDEM(searchEmbs.squeeze(0)[1].cpu().detach().numpy(), "search space layer 1")
        
        # self.displayDEM(polarEmb2.squeeze(0)[2].cpu().detach().numpy(), wait=False)
        # self.displayDEM(searchEmbs.squeeze(0)[2].cpu().detach().numpy(), "search space layer 2")

        # self.displayDEM(polarEmb2.squeeze(0)[3].cpu().detach().numpy())
        # self.displayDEM(searchEmbs.squeeze(0)[3].cpu().detach().numpy(), "search space layer 3")

        # if cv2.waitKey(0) == ord('q'):
        #     cv2.destroyAllWindows()

        ''' get 1. sizes
                2. create space for each angle bin
                3. angle list to multiply weights with to get a bin list
                4. vector to store scalars for each pair in the batch
        '''
        binAngles = torch.linspace(-np.pi, np.pi, emb1.shape[2] + 1).to(self.device)
        # binAngles = torch.linspace(0, 2*np.pi, emb1.shape[2] + 1).to(self.device)
        # f = torch.flip(binAngles, [0])
        # binAngles = torch.cat((binAngles, f))
        thetas = torch.zeros((batchSize, 1)).to(self.device)

        for batch in range(batchSize):
            searchSpace = searchEmbs[batch, ...].unsqueeze(0)
            kernel = polarEmb2[batch, ...].unsqueeze(0)
            angleWeights = F.softmax(F.conv2d(searchSpace, kernel), dim=2)

            # print(angleWeights)

            thetas[batch, 0] = torch.sum(torch.mul(angleWeights[:,0,:,0], binAngles)) - np.pi

        return thetas

    ''' take in thetas (batchSize, 1) and use each theta to rotate each emb (batchSize, 1, 125, 125)'''
    def transformEmb(self, emb, thetas):
        batchSize, _ = thetas.shape
        affineTransform = torch.zeros((batchSize, 2, 3)).to(self.device)

        s = torch.sin(thetas).to(self.device).T
        c = torch.cos(thetas).to(self.device).T

        sl1 = torch.vstack((c, -s)).unsqueeze(0)
        sl2 = torch.vstack((s, c)).unsqueeze(0)

        rotMatrices = torch.cat((sl1, sl2))

        rotMatrices = rotMatrices.swapaxes(0,2).swapaxes(2,1)

        affineTransform[:,:2,:2] = rotMatrices
        correlations = F.affine_grid(affineTransform, emb.shape, align_corners=True)
        transformedImages = F.grid_sample(emb, correlations, align_corners=True)

        return transformedImages

    def pixelwiseDist(self, emb1, emb2):
        batchSize, channelNum, h, w = emb1.size()

        arranged = emb1.reshape((batchSize, channelNum, h*w)).unsqueeze(2)
        arranged2 = emb2.reshape((batchSize, channelNum, h*w)).unsqueeze(2)

        rows = torch.tile(arranged, (1,1,h*w,1))
        cols = torch.tile(arranged2.swapaxes(-1,-2), (1,1,1,h*w))

        return torch.abs(rows - cols)


    ''' run the pipeline on two dems, dem1 is transformed after yawing
        return distance, reconstructed_dem1, reconstructed_dem2
    '''
    def forward(self, dem1, dem2):
        # print("Begun")
        
        from icecream import ic

        ''' get embeddeings '''
        emb1 = self.encoder(dem1)
        emb2 = self.encoder(dem2)


        ''' train enc-dec '''
        reconstructed_dem1 = self.decoder(emb1)
        reconstructed_dem2 = self.decoder(emb2)


        ''' get yaw and transform '''
        R = self.DYT(emb1, emb2)


        # self.displayDEM(emb1.squeeze(0)[0].cpu().detach().numpy(), "first DEM")
        # self.displayDEM(emb2.squeeze(0)[0].cpu().detach().numpy(), "second DEM")
        
        # print("DYT'ed")

        transformedEmb1 = self.transformEmb(emb1, R)


        # self.displayDEM(transformedEmb1.squeeze(0)[0].cpu().detach().numpy(), "transformed DEM")
        # if cv2.waitKey(0) == ord('q'):
        #     cv2.destroyAllWindows()

        # self.displayDEM(transformedEmb1.squeeze(0)[0].cpu().detach().numpy(), title="post-transform")

        # return None, None, None # TEMPORARY

        ''' enrich features, get the final distance '''
        emb1 = self.enricherCNN(transformedEmb1)
        emb2 = self.enricherCNN(emb2)


        pixelWiseMatrix = self.pixelwiseDist(emb1, emb2)
        

        # print("ebfigd: ", emb1.shape)
        # print(pixelWiseMatrix.shape)

        distance = self.distanceCNN(pixelWiseMatrix)


        return distance, R, reconstructed_dem1, reconstructed_dem2

''' encoder '''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        return x

''' decoder '''
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        # prevent negative DEM values
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))

        return x
    
'''final cnn to returrn a distance value given two embeddings'''
class DistanceCNN(nn.Module):
    def __init__(self):
        super(DistanceCNN, self).__init__()

        self.conv_1 = nn.Conv2d(10, 64, 5, padding=0, stride=2)
        self.conv_2 = nn.Conv2d(64, 32, 5, padding=0, stride=2)
        self.conv_3 = nn.Conv2d(32 , 4 , 1 , padding=0, stride=2) 
        self.fc1 = torch.nn.Linear(64516 , 100) #torch.nn.Linear(640, 100) # BUG
        self.fc2 = torch.nn.Linear(100, 10) 
        self.fc3 = torch.nn.Linear(10, 1) 
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.abs(self.fc3(x)) 

        return x 

''' block for the penultimate resnet '''
class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2 if self.downsample else 1),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.conv1(input))
        input = nn.ReLU()(self.conv2(input))
        input = nn.ReLU()(self.conv3(input))
        input = input + shortcut
        return nn.ReLU()(input)

''' penultimate cnn to enrich the embeddings '''
class EnricherCNN(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 10, 10, 128, 128] ## THIS

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
            self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
            self.layer2.add_module('conv3_%d' % (
                i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (
                i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,),resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)

        # input = self.layer2(input)
        # input = self.layer3(input)
        # input = self.layer4(input)
        # input = self.gap(input)
        
        # input = torch.flatten(input, start_dim=1)
        # input = self.fc(input)

        return input