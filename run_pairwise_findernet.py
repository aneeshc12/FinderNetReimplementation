# inference script to run findernet pairwise for all scans in a database, to calculate scores/loops

import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src2.myModels import end2endModel

with torch.no_grad():

    # args
    model_ckpt_path = '/home2/aneesh.chavan/FinderNetReimplementation/weights/cwt/cwt_weight.pt'
    DEM_root = '/scratch/aneesh.chavan/KITTI/09/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    end2end = end2endModel()
    end2end = end2end.to(device)

    end2end.load_state_dict(torch.load(model_ckpt_path) )
    print(" Model Loaded " + str(model_ckpt_path), flush=True)

    # define datasets and dataloaders
    class PairwiseKittiDataset(Dataset):
        def __init__(self, DEM_root):
            self.DEM_root = DEM_root

            self.file_names = sorted([os.path.join(DEM_root, f) for f in os.listdir(DEM_root)])[::3]
            self.length = len(self.file_names) ** 2

        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            idx1 = idx // len(self.file_names)
            idx2 = idx %  len(self.file_names)

            return self.file_names[idx1], self.file_names[idx2]
        
    dl = PairwiseKittiDataset(DEM_root)
    loader = DataLoader(dl, batch_size=24)
    print("dataloader created")

    # create dict to store scores and rotations
    score_dict = {}
    rot_dict = {}

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

    for anchor_DEM, query_DEM in tqdm(loader):
        # print(anchor_DEM)
        # print(query_DEM)

        if len(anchor_DEM) <= 0:
            continue

        anchor_imgs = readImg(anchor_DEM)
        query_imgs = readImg(query_DEM)

        anchor_imgs = anchor_imgs.unsqueeze(1)
        query_imgs = query_imgs.unsqueeze(1)

        score, R, _, _ = end2end(anchor_imgs, query_imgs)

        # print(score.shape)
        # print(R)

        for num, (i, j) in enumerate(zip(anchor_DEM, query_DEM)):
            if anchor_DEM[num] not in score_dict.keys():
                score_dict[anchor_DEM[num]] = {}

            if anchor_DEM[num] not in rot_dict.keys():
                rot_dict[anchor_DEM[num]] = {}

            score_dict[anchor_DEM[num]][query_DEM[num]] = score[num,0].detach().cpu()
            rot_dict[anchor_DEM[num]][query_DEM[num]] = R[num,0].detach().cpu()


    # save scores and rots
    # print(score_dict)
    # for i in rot_dict:
    #     print(i, rot_dict[i])
    
    import pickle 
    with open('./KITTI09_inference/score_dict.pkl', 'wb') as f:
        pickle.dump(score_dict, f)

    with open('./KITTI09_inference/rot_dict.pkl', 'wb') as f:
        pickle.dump(rot_dict, f)
