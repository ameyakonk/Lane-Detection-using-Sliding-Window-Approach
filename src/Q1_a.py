from turtle import colormode
import numpy as np
from numpy import linalg as LA
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image as im
from numpy.linalg import inv
from numpy import asarray
from argparse import ArgumentParser
import glob

class Histogram:

    def __init__(self):
        pass 

    def histEquilization(self):
        path = glob.glob("images/*.png")

        for p in path:
            img = cv2.imread(p, 0)
            
            row, col = img.shape
            newMat = np.ones([row,col,1],dtype=np.uint8)
            cvd = []
            cvd_coord = []
            coord_sum  = 0
            for i in range(256):
                coord = np.asarray(np.where(img == i))
                coord_sum += coord.shape[1]/img.size
                cvd_coord.append(coord)
                cvd.append(coord_sum)

            for i in range(256):
                newMat[cvd_coord[i][0], cvd_coord[i][1]] = 255*cvd[i]

            cv2.imshow("frame", newMat)
            
            if cv2.waitKey(10) & 0xff == ord('q'):
                cv2.destroyAllWindows()
        cv2.destroyAllWindows()
###############################################################################################################

h = Histogram()
h.histEquilization()
