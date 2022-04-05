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

    def advHistEquilization(self):
        def divideImg(img):
            windowsize_r = int(img.shape[1]/8)
            windowsize_c = int(img.shape[0]/8)
            windows = []
            for r in range(0,img.shape[0] - windowsize_c+1, windowsize_c):
                for c in range(0,img.shape[1] - windowsize_r+1, windowsize_r):
                    windows.append(img[r:r+windowsize_c,c:c+windowsize_r])
            return histogram(windows)

        def histogram(windows):
            clip = 40
            newImages = []
            for img in windows:
                row, col = img.shape
                newMat = np.ones([row,col,1],dtype=np.uint8)
                cvd = []
                pdf = []
                cvd_coord = []
                ##############################################
                clip_count = 0                      # count of excessive coordinates after clipping
                for i in range(256):
                    coord = np.asarray(np.where(img == i))
                    cvd_coord.append(coord)
                    pdf.append(clip if coord.shape[1] > clip else coord.shape[1])
                    clip_count += coord.shape[1]-clip if coord.shape[1] > clip else 0
                
                clip_count_add = int(clip_count/256)
                pdf = list(map(lambda x : x + clip_count_add, pdf))
                clip_count_rem = clip_count_add%256
                
                pdf = np.asarray(pdf)
                while(clip_count_rem !=0):
                    iter  = round(256/clip_count_rem)
                    clip_count_rem -= round(256/iter) if round(256/iter) <= clip_count_rem else clip_count_rem
                    for i in range(256, iter):
                        pdf[i] += 1
                pdf = np.divide(pdf, img.size)
                cvd = np.cumsum(pdf)
                for i in range(256):
                    newMat[cvd_coord[i][0], cvd_coord[i][1]] = round(255*cvd[i])
                newImages.append(newMat)

            return newImages
            
        ##################################################################################

        def reshapeImg(images, img):
            row, col = img.shape
            newImg = np.ones([row,col,1],dtype=np.uint8)
            windowsize_r = int(img.shape[1]/8)
            windowsize_c = int(img.shape[0]/8)
            windows = []
            i = 0
            for r in range(0,img.shape[0] - windowsize_c+1, windowsize_c):
                for c in range(0,img.shape[1] - windowsize_r+1, windowsize_r):
                    newImg[r:r+windowsize_c,c:c+windowsize_r] = images[i][0:windowsize_c,0:windowsize_r]
                    i += 1
            
            return newImg
        ##################################################################################
      
        path = glob.glob("images/*.png")
        
        temp = cv2.imread(path[1], 0)
        print(temp.shape)
       
        for p in path:
            img = cv2.imread(p, 0)
            newImg = reshapeImg(divideImg(img), img)
            cv2.imshow("frame", newImg)
            
            if cv2.waitKey(25) & 0xff == ord('q'):
                cv2.destroyAllWindows() 
        cv2.destroyAllWindows()

        ##################################################################################

##############################################################################################################       
h = Histogram()
h.advHistEquilization()
