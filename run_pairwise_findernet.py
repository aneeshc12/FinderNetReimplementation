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
    DEM_root = '/home2/aneesh.chavan/FinderNetReimplementation/inference_pcds/'
    DEM_list = '/home2/aneesh.chavan/FinderNetReimplementation/inference_pcds/anchor_DEMs'
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

            self.anchor_files = sorted([os.path.join(DEM_root, "anchor_DEMs", f) for f in os.listdir(os.path.join(DEM_root, "anchor_DEMs"))])
            self.query_files = sorted([os.path.join(DEM_root, "query_DEMs", f) for f in os.listdir(os.path.join(DEM_root, "query_DEMs"))])
            self.length = len(self.anchor_files) * len(self.query_files)

            print(len(self.anchor_files), len(self.query_files))

            print(self.anchor_files[:5])
            print()
            print(self.query_files[:5])

        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            idx1 = idx // len(self.anchor_files)
            idx2 = idx %  len(self.query_files)

            return self.anchor_files[idx1], self.query_files[idx2]
        
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
    with open('/home2/aneesh.chavan/FinderNetReimplementation/inference_pcds/inference_scores/cross_score_dict.pkl', 'wb') as f:
        pickle.dump(score_dict, f)

    with open('/home2/aneesh.chavan/FinderNetReimplementation/inference_pcds/inference_scores/cross_rot_dict.pkl', 'wb') as f:
        pickle.dump(rot_dict, f)
