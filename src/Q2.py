from pickletools import uint8
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

class LaneDetection:
    def __init__(self):
        self.clip = 330
        pass

    def readVideo(self):
        cap = cv2.VideoCapture('whiteline.mp4')
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                count += 1
                cv2.imshow("frame", frame)
                cv2.imwrite("q2images/frame%d.jpg" % count, frame)     # save frame as JPEG file   
                if cv2.waitKey(25) & 0xff == ord('q'):
                    cv2.destroyAllWindows()

    def laneDetect(self):

        def findDistance(x1, y1, x2, y2):
            return (x1-x2)**2 + (y1-y2)**2
        
        cap = cv2.VideoCapture('whiteline.mp4')
        count = 0
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret == True:
                #img = cv2.imread("frame1.jpg")
                print(img.shape)
                img1 = img.copy()
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + 
                                                     cv2.THRESH_OTSU)
                rows = img.shape[0]
                cols = img.shape[1]

                #### clockwise - 1:(120,535), 2:(350,370), 3:(620,370), 4:(900,535)
                #### heights - 1:(120,535), (350,370) 2:(620,370), (900,535)
                #### widths - 1:(120,535),(900,535) 2:(620,370),(350,370)

                cv2.line(gray,(120,535),(900,535),(0,0,255),2)    
                cv2.line(gray,(120,535),(350,370),(0,0,255),2)    
                cv2.line(gray,(620,370),(350,370),(0,0,255),2)    
                cv2.line(gray,(620,370),(900,535),(0,0,255),2)    
                
                pt_A = (120,535)
                pt_B = (350,370)
                pt_C = (620,370)
                pt_D = (900,535)
                
                stencil = np.zeros_like(gray[:, :])
                polygon = np.array([pt_A, pt_B, pt_C, pt_D])
                cv2.fillConvexPoly(stencil, polygon, 1)
            
                gray = cv2.bitwise_and(gray, gray,mask=stencil)
             
                ret, thresh = cv2.threshold(gray, 130, 145, cv2.THRESH_BINARY)
               
                lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 50, 50, maxLineGap=50)
                print(len(lines))
               
                
                cleans = np.empty(shape=[0,4], dtype=np.int32)
                for l in lines:
                    alfa = math.degrees(math.atan2(l[0][2]-l[0][0], l[0][3]-l[0][1]))

                    if len(cleans) == 0:
                        cleans = np.append(cleans, [l[0]], axis=0)
                        continue

                    similar = False
                    for c in cleans:
                        beta = math.degrees(math.atan2(c[2]-c[0], c[3]-c[1]))
                        if abs(alfa-beta) <= 15:
                            similar = True
                            break

                    if not similar:
                        cleans = np.append(cleans, [l[0]], axis=0)

          
                if(len(cleans)<2): 
                    continue
                min_val = 10000000
                max_val = 0
                min_ = 0
                max_ = 0
                for line in [cleans]:
                    for x1,y1,x2,y2 in line:
                        if findDistance(x1.item(), y1.item(), x2.item(), y2.item()) < min_val:
                            min_val = findDistance(x1.item(), y1.item(), x2.item(), y2.item())
                            min_ = (x1, y1, x2, y2)
                        
                        if findDistance(x1.item(), y1.item(), x2.item(), y2.item()) >= max_val:
                            max_val = findDistance(x1.item(), y1.item(), x2.item(), y2.item())
                            max_ = (x1, y1, x2, y2)

                print(min_[0])            
                cv2.line(img,(min_[0],min_[1]),(min_[2],min_[3]),(0, 0, 255),5)
                cv2.line(img,(max_[0],max_[1]),(max_[2],max_[3]),(0, 255, 0),5)
                
                count += 1
                cv2.imshow("frame", img) 
        
                if cv2.waitKey(25) & 0xff == ord('q'):
                    cv2.destroyAllWindows()

l = LaneDetection()
l.laneDetect()