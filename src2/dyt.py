import numpy as np
import cv2
from tqdm import tqdm

from generateDEM import displayDEM

def convertToPolar(DEM="../sampleKittiData/DEMs/000010.npy"):
    DEM = np.load(DEM)
    h, w = np.shape(DEM)

    displayDEM(DEM)

    # may need adjustment
    pW = 180
    pH = int(np.sqrt((w/2)**2 + (h/2)**2)) + 1
    polarDEM = np.zeros((pH, pW))

    # def getPolarCoords(x1, x2):
    #     pY = np.sqrt(x1**2 + x2**2)
        
    #     # get x1, convert all values to -pi,pi
    #     if x1 != 0:
    #         if x2 != 0:
    #             pX = np.arctan(x2/x1)
    #     else:
    #         if x2 >= 0:
    #             pX = np.pi/2
    #         else:
    #             pX = 3*np.pi/2

    #     if pX

    # start at the top left corner
    for Y in range(h):
        for X in range(w):
            # get x1 and x2
            x1 = X - w/2
            x2 = Y - h/2

            # fill in the polar image
            pX = 360/(2*np.pi) * (np.arctan(x2/x1) if x1 != 0 else (1 if x2 >= 0 else -1) * np.pi/2)
            pY = np.sqrt(x1**2 + x2**2)

            pX = int(pX)
            pY = int(pY)

            # normalise
            print("px: ", pX, " | py: ", pY)

            polarDEM[pY, pX] = DEM[Y, X]

    displayDEM(polarDEM)

convertToPolar()