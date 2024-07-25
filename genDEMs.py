from src.generateDEM import * 
import os
import sys
import numpy as np
from tqdm import tqdm

def genFolder(dirNum):
    dirName = f'/scratch/aneesh/06/velodyne/'
    outputName = f'./kitti_dem_data/'
    files = os.listdir(dirName)

    for i, file in enumerate(files):
        f = os.path.join(dirName, file)
        num = file.split('.')[0]
        k = generateDEM(f)
        print(f"%d  %d/%d  %s: " % (dirNum, i, len(files), num) ,f, end=' ')

        if type(k) != None:
            # np.save(os.path.join(outputName, num), k)
            # print(k)
            # displayDEM(k)
            # k = np.array(k * 256, dtype=np.uint8)
            cv2.imwrite(os.path.join(outputName, num + '.png'), k)
            print(" - succesful!")
        else:
            print(" - Error!!")

genFolder(1)
