from src2.generateDEM import * 
import os
import sys
import numpy as np
from tqdm import tqdm

def write_pose_in_kitti_format(arr):
    final = ""

    if len(arr.shape) != 1:
        arr = arr.reshape(-1)
    for i in arr[:-1]:
        final += str(i) + ", "
    
    final += str(arr[-1]) + "\n"
    return final

def genFolder(dirNum):
    # dirName = f'/scratch/aneesh/06/velodyne/'
    dirName = f'/scratch/aneesh.chavan/KITTI_raw/00/velodyne/'
    outputName = f'./'
    files = os.listdir(dirName)

    with open(os.path.join(outputName, "final_transforms.txt"), 'w') as pose_file:
        for i, file in enumerate(files):
            f = os.path.join(dirName, file)
            num = file.split('.')[0]
            k, final_transform = generateDEM(f)
            print(f"%d  %d/%d  %s: " % (dirNum, i, len(files), num) ,f, end=' ')

            pose_file.write(write_pose_in_kitti_format(final_transform))

            if type(k) != None:
                # np.save(os.path.join(outputName, num), k)
                # print(k)
                # displayDEM(k)
                k = np.array(k * 256, dtype=np.uint8)
                cv2.imwrite(os.path.join(outputName, num + '.png'), k)
                print(" - succesful!")
            else:
                print(" - Error!!")

genFolder(1)
