# inference script to run findernet pairwise for all scans in a database, to calculate scores/loops

import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import csv
from src2.myModels import end2endModel

with torch.no_grad():

    # args
    model_ckpt_path = '/home2/aneesh.chavan/FinderNetReimplementation/weights/cwt/cwt_weight.pt'
    DEM_root = '/scratch/aneesh.chavan/KITTI/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    csv_path = '/home2/aneesh.chavan/FinderNetReimplementation/src2/TrainKitti.csv'

    # load model
    end2end = end2endModel()
    end2end = end2end.to(device)

    end2end.load_state_dict(torch.load(model_ckpt_path) )
    print(" Model Loaded " + str(model_ckpt_path), flush=True)

    def read_csv_triplets(file_path, start_row):
        triplets = []
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            # Skip rows up to the start_row (1-based index)
            for _ in range(start_row - 1):
                next(reader)
            # Read the remaining rows
            for row in reader:
                triplets.append(tuple([os.path.join(DEM_root, i) for i in row]))
                print(tuple([os.path.join(DEM_root, i) for i in row]))
        return triplets

        # run the model pairwise for all images
    def readImg(paths):
        imgs = []
        for i in range(len(paths)):
            img = np.asarray(Image.open(paths[i])).astype(np.float32)
            # img = np.load(paths[i]).astype(np.float32)
            img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)

            # imgs.append(img.astype(np.uint8))
            imgs.append(img)
        
        imgs = torch.tensor(np.array(imgs), dtype=torch.float).to(device)
        return imgs
    
    def readOneImg(paths):
        imgs = []
        for i in range(1):
            img = np.asarray(Image.open(paths)).astype(np.float32)
            # img = np.load(paths[i]).astype(np.float32)
            img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)

            # imgs.append(img.astype(np.uint8))
            imgs.append(img)
        
        imgs = torch.tensor(np.array(imgs), dtype=torch.float).to(device)
        return imgs

    # define datasets and dataloaders
    class PairwiseKittiDataset(Dataset):
        def __init__(self, triplets):
            self.triplets = triplets

            self.pairs = []
            for t in self.triplets:
                self.pairs.append([t[0], t[1]])
                self.pairs.append([t[0], t[2]])

        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            pair = self.pairs[idx]

            anchor_DEM = readOneImg(pair[0])
            query_DEM = readOneImg(pair[1])

            return anchor_DEM, query_DEM, pair
    
    triplets = read_csv_triplets(csv_path, 14607)
    dl = PairwiseKittiDataset(triplets)
    loader = DataLoader(dl, batch_size=24)
    print("dataloader created")

    # create dict to store scores and rotations
    score_dict = {}
    rot_dict = {}


    for anchor_DEM, query_DEM, pairs in tqdm(loader):
        # print(anchor_DEM)
        # print(query_DEM)

        if len(anchor_DEM) <= 0:
            continue

        anchor_imgs = anchor_DEM
        query_imgs = query_DEM

        # print(anchor_imgs.shape, query_imgs.shape)

        # print(pairs)

        score, R, _, _ = end2end(anchor_imgs, query_imgs)

        # print(score.shape)
        # print(R)

        for num, (i, j) in enumerate(zip(pairs[0], pairs[1])):
            # a_key = int(i.split('/')[-1].split('.')[0])
            # q_key = int(j.split('/')[-1].split('.')[0])
            a_key = i
            q_key = j

            if a_key not in score_dict.keys():
                score_dict[a_key] = {}

            if a_key not in rot_dict.keys():
                rot_dict[a_key] = {}

            score_dict[a_key][q_key] = score[num,0].detach().cpu()
            rot_dict[a_key][q_key] = R[num,0].detach().cpu()


    # save scores and rots
    # print(score_dict)
    # for i in rot_dict:
    #     print(i, rot_dict[i])
    
    import pickle 
    with open('./KITTI09_inference/score_dict.pkl', 'wb') as f:
        pickle.dump(score_dict, f)

    with open('./KITTI09_inference/rot_dict.pkl', 'wb') as f:
        pickle.dump(rot_dict, f)
