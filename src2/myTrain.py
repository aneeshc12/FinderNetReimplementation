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
from natsort import natsorted
from math import log2
import sys
from tqdm import tqdm
from scipy import ndimage
from icecream import ic

from myModels import *

from generateDEM import displayDEM

from PIL import Image

import wandb

parser = argparse.ArgumentParser(description='Train Code for Spatial Transformer')
# parser.add_argument('--data_path', help='Path to the dataset', default='/home2/aneesh.chavan/FinderNetReimplemantation/src2/TrainKitti.csv')
parser.add_argument('--data_path', help='Path to the dataset', default='/home2/aneesh.chavan/FinderNetReimplemantation/src2/TrainLuF.csv')
parser.add_argument('--base_path', help='path to parent directory of the image dataset folder', default='/scratch/aneesh.chavan/LuF')
parser.add_argument('--batch_size', help='Size of Batch', default= 12)
parser.add_argument('--lr', help='Learning rate', default= 1e-4)
parser.add_argument('--num_epochs', help='Number of epochs', default = 30 )
parser.add_argument('--image_resolution', help='Size of image ', default =500)
parser.add_argument('--save_path', help='base path to save the model', default='/home2/aneesh.chavan/FinderNetReimplemantation/weights/LuF')
parser.add_argument('--iters_per_ckpt', help= 'number of iterations to save a checkpoint', default=3)
# parser.add_argument('--total_train_samples', help= 'Total number of train samples', default= 12)            # 14606
parser.add_argument('--total_train_samples', help= 'Total number of train samples', default= 16000)            # 14606
parser.add_argument('--total_test_samples', help='Total number of validation/test samples', default= 6000)  # 
parser.add_argument('--start_index', help='enter row number of the csv to consider as start ', default=0)
parser.add_argument('--margin', help ='margin of the triplet loss ', default= 0.5 )
parser.add_argument('--continue_train', help =' Continue traiing from a previous checkpoint ', default= False )
parser.add_argument('--path_to_prev_ckpt', help =' path to the previous checkpoint only required if continue_train is true ', default= None )
parser.add_argument('--lr_change_frequency', help='Number of epochs to update the learning rate', default=10)
parser.add_argument('--grad_clip_norm', help='grad_clip_norm', default=1)

args = parser.parse_args()


''' return a minibatch containing each image stores at the paths in `paths` '''
def readImg(paths):
    imgs = []
    for i in range(len(paths)):
        # img = np.asarray(Image.open(paths[i])).astype(np.float32)
        img = np.load(paths[i]).astype(np.float32)
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)

        # imgs.append(img.astype(np.uint8))
        imgs.append(img)
    
    imgs = torch.tensor(np.array(imgs), dtype=torch.float).to(device)
    return imgs

''' torch init '''
torch.autograd.set_detect_anomaly(True)
SEED = 123456789
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True ## makes the process deterministic i.e when a same input is given to a the same algortihm on the same hardware the result is the same
torch.backends.cudnn.benchmark = False ## It enables auto tune to find the best algorithm for the given hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

"""WandB init"""

wandb.init(
    # set the wandb project where this run will be logged
    project="FinderNet reimplementation",

    # track hyperparameters and run metadata
    config={
    "batch_size": args.batch_size,
    "epochs": args.num_epochs,
    "lr": args.lr,
    }
)



# print("Total samples: " + str(args.total_train_samples), flush=True)
numBatches = math.ceil(int(args.total_train_samples)/int(args.batch_size))
numValBatches = math.ceil(int(args.total_test_samples/int(args.batch_size)))

def CreateBatchData(file, start_index=0, mode='train'):
    if(mode == 'train'):
        final = int(args.total_train_samples)
    if(mode == 'validation'):
        start_index = args.total_train_samples
        final = args.total_test_samples + args.total_train_samples

    EndIndex = start_index + final #args.total_train_samples

    # anchorSamples = DF['anchor'][start_index:EndIndex]
    # positiveSamples = DF['positive'][start_index:EndIndex]
    # negativeSamples = DF['negative'][start_index:EndIndex]

    f = open(file, 'r')
    demLists = []
    for l in f.readlines():
        demLists.append(l.split(','))
        demLists[-1][2] = demLists[-1][2][:-1]
    demLists = np.array(demLists)[1:]
    f.close()

    anchorSamples = list(demLists[start_index : start_index + final, 0])
    positiveSamples = list(demLists[start_index : start_index + final, 1])
    negativeSamples = list(demLists[start_index : start_index + final, 2])

    anchorDataSet = []
    positiveDataSet = []
    negativeDataSet = []
    
    for i in range(start_index, start_index + final, args.batch_size):
        # print("From ", i, " to ", min(len(positiveSamples), i + args.batch_size))

        a = anchorSamples[i : min(len(anchorSamples), i + args.batch_size)]
        p = positiveSamples[i : min(len(positiveSamples), i + args.batch_size)]
        n = negativeSamples[i : min(len(negativeSamples), i + args.batch_size)]

        for k, (x, y, z) in enumerate(zip(a, p, n)):
            a[k] = os.path.join(args.base_path, x)
            p[k] = os.path.join(args.base_path, y)
            n[k] = os.path.join(args.base_path, z)

        anchorDataSet.append(a)
        positiveDataSet.append(p)
        negativeDataSet.append(n)

    return anchorDataSet, positiveDataSet, negativeDataSet

end2end = end2endModel()

# n = readImg(anchorDS[10])
# print(anchorDS[10])

# a = np.load(anchorDS[0][0])
# a = cv2.resize(a, (500, 500), interpolation=cv2.INTER_NEAREST)
# plt.imshow(a)
# plt.show()


# plt.imshow(n[0].detach().cpu())
# plt.show()

# displayDEM(np.array(n[0].detach().cpu()))


mseLoss = nn.MSELoss()

if(args.continue_train):
    end2end.load_state_dict(torch.load(args.path_to_prev_ckpt) )
    print(" Model Loaded " + str(args.path_to_prev_ckpt), flush=True)
    
print("Model init'ed")

'''
Init loading, scheduler optimizer etc.
'''
end2end.to(device)

optimizer = optim.AdamW(end2end.parameters(), lr=args.lr)
save_checkpoint = 0
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_change_frequency, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

zeros = torch.zeros((args.batch_size, 1)).to("cuda")

saveCounter = 0

'''create datasets containing minibatches of size 12'''
# anchorDataSet, positiveDataSet, negativeDataSet = CreateBatchData('/home2/aneesh.chavan/FinderNetReimplemantation/src2/TrainKitti.csv')
# val_anchorDataSet, val_positiveDataSet, val_negativeDataSet = CreateBatchData('/home2/aneesh.chavan/FinderNetReimplemantation/src2/TrainKitti.csv', mode='validation')

anchorDataSet, positiveDataSet, negativeDataSet = CreateBatchData('/home2/aneesh.chavan/FinderNetReimplemantation/src2/TrainLuF.csv')
val_anchorDataSet, val_positiveDataSet, val_negativeDataSet = CreateBatchData('/home2/aneesh.chavan/FinderNetReimplemantation/src2/TrainLuF.csv', mode='validation')

print("Datasets created")

ic(numBatches)
print("Beginning training")

for e in range(args.num_epochs):
    epochLoss = 0
    correct = 0
    validatonCorrect = 0
    epoch_classification_loss = 0
    epoch_reconstruction_loss = 0

    '''iterate over all batches'''
    for batch_num in tqdm(range(numBatches)):
    # for batch_num in range(numBatches):
        optimizer.zero_grad()
        if(len(anchorDataSet[batch_num]) <= 0):
            continue
        
        anchorImgs = readImg(anchorDataSet[batch_num])
        positiveImgs = readImg(positiveDataSet[batch_num])
        negativeImgs = readImg(negativeDataSet[batch_num])
    
        # for i in range(args.batch_size):
        #     print("triplet ", i)
            # displayDEM(anchorImgs[i].detach().cpu())
            # displayDEM(positiveImgs[i].detach().cpu())
            # displayDEM(negativeImgs[i].detach().cpu())
    
        '''images need a channel, make Bx250x250 -> Bx1x250x250s'''
        anchorImgs = anchorImgs.unsqueeze(1)
        positiveImgs = positiveImgs.unsqueeze(1)
        negativeImgs = negativeImgs.unsqueeze(1)
    
        '''Recover anchor, positive and negative embs. Also get reconstructed pcds, to compute enc-dec loss'''
        '''Compute distance between anchor-positive / anchor-negative embs, feed this to loss laster'''
        scores_ap, _, reconstructed_DEM_a1, reconstructed_DEM_p = end2end.forward(anchorImgs, positiveImgs)
        scores_an, _, reconstructed_DEM_a2, reconstructed_DEM_n = end2end.forward(anchorImgs, negativeImgs)
        
        '''triplet loss of distance scores'''
        # print("apshape!: ", scores_ap.shape)
        # print("anshape!: ", scores_ap.shape)
    
        # print(", argin:.", args.margin)
        # print(", argin:.", zeros)
    
        zeros = torch.zeros((scores_ap.shape[0], 1)).to("cuda")
        classification_loss = torch.sum(torch.maximum(scores_ap - scores_an + args.margin, zeros).to(device))
        
        '''dem enc-dec loss'''
        reconstruction_loss = (mseLoss(reconstructed_DEM_a1, anchorImgs ) + mseLoss(reconstructed_DEM_a2, anchorImgs) 
                                + mseLoss(reconstructed_DEM_p, positiveImgs ) + mseLoss(reconstructed_DEM_n, negativeImgs))
    
        epoch_classification_loss += classification_loss.item()
        epoch_reconstruction_loss += 0.1*reconstruction_loss.item()
    
        '''backprop'''
        # loss = classification_loss + 0.1*reconstruction_loss 
        loss = classification_loss + 0.1 * reconstruction_loss 
        # loss = reconstruction_loss
        # print(reconstruction_loss)
        # loss = reconstruction_loss
        
        loss.backward()
        # print("grad: ", torch.autograd.grad(reconstruction_loss, reconstructed_DEM_a1))
        
        # torch.nn.utils.clip_grad_norm_(end2end.parameters(), max_norm=args.grad_clip_norm)

        optimizer.step()
        epochLoss += loss.item()
    
        # sanity check
        with torch.no_grad():
            random_rec_loss = (mseLoss(anchorImgs.max() * torch.rand_like(reconstructed_DEM_a1), anchorImgs.max() * torch.rand_like(anchorImgs)) 
                            + mseLoss(anchorImgs.max() * torch.rand_like(reconstructed_DEM_a2), anchorImgs.max() * torch.rand_like(anchorImgs)) 
                            + mseLoss(anchorImgs.max() * torch.rand_like(reconstructed_DEM_p),  anchorImgs.max() * torch.rand_like(positiveImgs)) 
                            + mseLoss(anchorImgs.max() * torch.rand_like(reconstructed_DEM_n),  anchorImgs.max() * torch.rand_like(negativeImgs)))
            # ic(random_rec_loss)


    scheduler.step(epochLoss)

    val_epochLoss = 0
    correct = 0
    validatonCorrect = 0
    val_epoch_classification_loss = 0
    val_epoch_reconstruction_loss = 0
    # # run validation loop
    # with torch.no_grad():
    #     '''iterate over all batches'''
    #     for batch_num in tqdm(range(numValBatches)):
    #     # for batch_num in range(numBatches):
    #         if(len(anchorDataSet[batch_num]) <= 0):
    #             continue
            
    #         anchorImgs = readImg(val_anchorDataSet[batch_num])
    #         positiveImgs = readImg(val_positiveDataSet[batch_num])
    #         negativeImgs = readImg(val_negativeDataSet[batch_num])
        
    #         # for i in range(args.batch_size):
    #         #     print("triplet ", i)
    #             # displayDEM(anchorImgs[i].detach().cpu())
    #             # displayDEM(positiveImgs[i].detach().cpu())
    #             # displayDEM(negativeImgs[i].detach().cpu())
        
    #         '''images need a channel, make Bx250x250 -> Bx1x250x250s'''
    #         anchorImgs = anchorImgs.unsqueeze(1)
    #         positiveImgs = positiveImgs.unsqueeze(1)
    #         negativeImgs = negativeImgs.unsqueeze(1)
        
    #         '''Recover anchor, positive and negative embs. Also get reconstructed pcds, to compute enc-dec loss'''
    #         '''Compute distance between anchor-positive / anchor-negative embs, feed this to loss laster'''
    #         scores_ap, _, reconstructed_DEM_a1, reconstructed_DEM_p = end2end.forward(anchorImgs, positiveImgs)
    #         scores_an, _, reconstructed_DEM_a2, reconstructed_DEM_n = end2end.forward(anchorImgs, negativeImgs)
            
        
    #         zeros = torch.zeros((scores_ap.shape[0], 1)).to("cuda")
    #         classification_loss = torch.sum(torch.maximum(scores_ap - scores_an + args.margin, zeros).to(device))
            
    #         '''dem enc-dec loss'''
    #         reconstruction_loss = (mseLoss(reconstructed_DEM_a1, anchorImgs ) + mseLoss(reconstructed_DEM_a2, anchorImgs) 
    #                                 + mseLoss(reconstructed_DEM_p, positiveImgs ) + mseLoss(reconstructed_DEM_n, negativeImgs))
        
    #         val_epoch_classification_loss += classification_loss.item()
    #         val_epoch_reconstruction_loss += 0.1*reconstruction_loss.item()
        
    #         loss = classification_loss + 0.1 * reconstruction_loss 
    #         val_epochLoss += loss.item()

    epoch_classification_loss /= numBatches
    epoch_reconstruction_loss /= numBatches
    val_epoch_classification_loss /= numValBatches
    val_epoch_reconstruction_loss /= numValBatches

    wandb.log({
        "classification_loss": epoch_classification_loss, 
        "reconstruction_loss": epoch_reconstruction_loss,
        "val_classification_loss": val_epoch_classification_loss, 
        "val_reconstruction_loss": val_epoch_reconstruction_loss
    })

    saveCounter += 1
    print(f"Epoch {e}/{args.num_epochs}")
    print("Epoch loss = "  + str(epochLoss))
    print("Classification = " + str(epoch_classification_loss))
    print("Reconstruction = " + str(epoch_reconstruction_loss))
    print("val Classification = " + str(val_epoch_classification_loss))
    print("val Reconstruction = " + str(val_epoch_reconstruction_loss), flush=True )
    if(saveCounter % args.iters_per_ckpt == 0):
        save_file_name = "model_" + "epoch_num_" + str(e) + "_Loss_" + str(epochLoss) + ".pt"
        save_file_name = os.path.join(args.save_path, save_file_name)
        torch.save(end2end.state_dict(), save_file_name)

wandb.finish()