from src2.generateDEM import * 
import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def decompose_transformation_matrix(matrix):
    # Ensure the matrix is a numpy array
    matrix = np.array(matrix)

    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")

    # Extract the rotation matrix (upper-left 3x3 part)
    rotation_matrix = matrix[:3, :3]
    
    # Extract the translation vector (upper-right 3x1 part)
    translation = matrix[:3, 3]

    # Create a Rotation object from the rotation matrix
    rotation = R.from_matrix(rotation_matrix).as_quat()

    return rotation, translation

def write_pose_in_kitti_format(arr, num):
    final = f"{num}, "

    q, t = decompose_transformation_matrix(arr)

    # if len(arr.shape) != 1:
    #     arr = arr.reshape(-1)
    for i in t:
        final += str(i) + ", "
    for i in q[:-1]:
        final += str(i) + ", "

    final += str(q[-1]) + "\n"
    return final

def genFolder(dirNum):
    # dirName = f'/scratch/aneesh/06/velodyne/'
    dirName = f'/scratch/aneesh.chavan/KITTI_raw/{dirNum}/velodyne/'
    outputName = f'/scratch/aneesh.chavan/myKITTI_DEMS/{dirNum}'
    files = os.listdir(dirName)

    pose_list = []
    with open(os.path.join(outputName, "final_transforms.txt"), 'w') as pose_file:
        for i, file in enumerate(files):
            f = os.path.join(dirName, file)
            num = file.split('.')[0]
            k, final_transform = generateDEM(f)
            print(f"%s  %d/%d  %s: " % (dirNum, i, len(files), num) ,f, end=' ')

            pose_file.write(write_pose_in_kitti_format(final_transform, num))

            # print(write_pose_in_kitti_format(final_transform))
            # exit(0)

            if type(k) != None:
                # np.save(os.path.join(outputName, num), k)
                # print(k)
                # displayDEM(k)
                k = np.array(k * 256, dtype=np.uint8)
                cv2.imwrite(os.path.join(outputName, num + '.png'), k)
                print(" - succesful!")
            else:
                print(" - Error!!")

genFolder("00")
genFolder("05")
genFolder("06")
genFolder("07")
# genFolder("09")
