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



cap = cv2.VideoCapture('challenge.mp4')
count = 0
prev_left= []
prev_right = []

while(cap.isOpened()):
    ret, img = cap.read()
    org_img = img.copy()
    if ret == True:
      
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_scale = gray.copy()
        ret, thresh_temp = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + 
                                        cv2.THRESH_OTSU)
    
        thresh_temp = cv2.cvtColor(thresh_temp,cv2.COLOR_GRAY2BGR)

        rows = img.shape[0]
        cols = img.shape[1]

        #### clockwise - 1:(120,535), 2:(350,370), 3:(620,370), 4:(900,535)
        #### heights - 1:(120,535), (350,370) 2:(620,370), (900,535)
        #### widths - 1:(120,535),(900,535) 2:(620,370),(350,370)

        pt_A = (230,670)
        pt_B = (570,450)
        pt_C = (750,450)
        pt_D = (1110,670)

        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, maxHeight - 1],
                        [0, 0],
                        [maxWidth - 1, 0]
                        ,[maxWidth - 1, maxHeight- 1]])

        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        out = cv2.warpPerspective(gray,M,(maxWidth, maxHeight))
       # Minv = cv2.getPerspectiveTransform(output_pts,input_pts)
        Minv = np.linalg.inv(M)
        out_ = cv2.warpPerspective(gray,Minv,(maxWidth, maxHeight))

       
        ret, thresh = cv2.threshold(out, 120, 255, cv2.THRESH_BINARY + 
                                        cv2.THRESH_OTSU)


        histogram = np.sum(thresh[thresh.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])                    #points to the x coordinate to the left of the midpoint
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint        #points to the y coordinate to the right of the midpoint

        left_cur_x = leftx_base.copy()
        right_cur_x = rightx_base.copy()

        n = 9
        div = int(thresh.shape[0]/n) 
        margin = 50
        minpixels = 50
      
        thresh1 =  thresh.copy()
        thresh1 = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2BGR)

        if(len(thresh.nonzero()[0]) < 100000):
           
            left_indices = []
            right_indices = []
            new_left = []
            new_right = []
        
            for i in range(n):
                y_min = thresh.shape[0] - (i+1)*div
                y_max = thresh.shape[0] - (i)*div
                left_x_min = left_cur_x - margin
                left_x_max = left_cur_x + margin
                right_x_min = right_cur_x - margin
                right_x_max = right_cur_x + margin

                stencil_left = np.zeros_like(thresh[:, :])
                stencil_right = stencil_left.copy()

                cv2.rectangle(stencil_left,(left_x_min,y_min),(left_x_max,y_max),255, -1)
                cv2.rectangle(stencil_right,(right_x_min,y_min),(right_x_max,y_max),255, -1)

                gray_left = cv2.bitwise_and(thresh, thresh,mask=stencil_left)
                gray_right = cv2.bitwise_and(thresh, thresh,mask=stencil_right)

                if gray_left.nonzero()[1].size > minpixels:
                    left_cur_x = int(np.mean(gray_left.nonzero()[1]))
                    left_cur_y = int(np.mean(gray_left.nonzero()[0]))
                    left_indices.append([left_cur_x, left_cur_y])
                    left_np_array = np.asarray(left_indices)

                if gray_right.nonzero()[1].size > minpixels:
                    right_cur_x = int(np.mean(gray_right.nonzero()[1]))
                    right_cur_y = int(np.mean(gray_right.nonzero()[0]))
                    right_indices.append([right_cur_x, right_cur_y])
                    right_np_array = np.asarray(right_indices)

                cv2.rectangle(thresh1,(left_x_min,y_min),(left_x_max,y_max),(0,255,0), 2)
                cv2.rectangle(thresh1,(right_x_min,y_min),(right_x_max,y_max),(0,255,0), 2)

            pts_left = left_np_array.reshape((-1, 1, 2))
            pts_right = right_np_array.reshape((-1, 1, 2))
        else :
                pts_left = prev_left
                pts_right = prev_right
        
       
        thresh1 = cv2.polylines(thresh1, [pts_left], 
                       False, (255, 0, 0), 10)
        thresh1 = cv2.polylines(thresh1, [pts_right], 
                       False, (0, 255, 0), 10)

        _, IM = cv2.invert(M)
        
        leftx = []
        lefty = []
        for coord in left_np_array:
            coord = [coord[0], coord[1], 1]
            P = np.float32(coord)
            x, y, z = np.dot(IM, P)
            x = int(x/z)
            y = int(y/z)
            leftx.append(x)
            lefty.append(y)
            new_left.append([x, y])   

        new_np_left = np.asarray(new_left)

        rightx = []
        righty = []
        for coord in right_np_array:
            coord = [coord[0], coord[1], 1]
            P = np.float32(coord)
            x, y, z = np.dot(IM, P)
            x = int(x/z)
            y = int(y/z)
            rightx.append(x)
            righty.append(y)
            new_right.append([x, y])
            
        new_np_right = np.asarray(new_right)
        
        new_np_left = new_np_left.reshape((-1, 1, 2))
        new_np_right = new_np_right.reshape((-1, 1, 2))
        
        gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
   
        if not len(leftx) is 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            lc,res,_,_,_ = np.polyfit(lefty,leftx,2,full = True)
            lfit = np.poly1d(lc)

        if not len(rightx) is 0:
            right_fit = np.polyfit(righty, rightx, 2)
            rc,res,_,_,_ = np.polyfit(righty,rightx,2,full = True)
            rfit = np.poly1d(rc)
        
        ploty = np.linspace(gray.shape[0]//2 + 100, gray.shape[0]-50, gray.shape[0]//2)

        if not len(leftx) is 0:
            y_eval = gray.shape[0]
            ym_per_pix = 5/gray.shape[0]
            xm_per_pix = 0.3/gray.shape[1]
            left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
            right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        if not len(leftx) is 0:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            pts_left_ = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right_ = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

            pts = np.hstack((pts_left_, pts_right_))

            color_warp = np.zeros((gray.shape[0], gray.shape[1], 3), dtype='uint8')

            gray = cv2.polylines(org_img, np.int_([pts_left_]), 
                       False, (255, 0, 0), 5)
            gray = cv2.polylines(org_img, np.int_([pts_right_]), 
                       False, (0, 0, 255), 5)

            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
            org_img = cv2.addWeighted(org_img, 1, color_warp, 0.3, 0)
            avg_curve = ((left_curverad + right_curverad)/2)/100
            label_str = 'Radius of curvature: %.1f m' % avg_curve
            org_img = cv2.putText(org_img, label_str, (30,80), 0, 1, (255,0,0), 2, cv2.LINE_AA)
            if avg_curve < 40:
                turn = 'Go Straight' 
            else :
                turn = 'Turn Right' 
            org_img = cv2.putText(org_img, turn, (30,120), 0, 1, (255,0,0), 2, cv2.LINE_AA)
      
        prev_left = pts_left
        prev_right = pts_right
       
        scale_percent = 40 # percent of original size
        width = int(thresh1.shape[1] * scale_percent / 100)
        height = int(thresh1.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        stencil = np.zeros((gray.shape[0]-4*height, width, 3), np.uint8)
        gray_scale = cv2.cvtColor(gray_scale,cv2.COLOR_GRAY2BGR)
       
        # resize image
        thresh1 = cv2.resize(thresh1, dim, interpolation = cv2.INTER_AREA)
        thresh_temp = cv2.resize(thresh_temp, dim, interpolation = cv2.INTER_AREA)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        gray_scale = cv2.resize(gray_scale, dim, interpolation = cv2.INTER_AREA)
       
        a = np.concatenate((img, gray_scale), axis = 0)
        b = np.concatenate((thresh_temp, thresh1), axis = 0)
        c = np.concatenate((a, b), axis = 0)
        d = np.concatenate((c, stencil), axis = 0)
        main = np.concatenate((org_img, d), axis = 1)

        label_str = '1. Lane and radius detection: '
        main = cv2.putText(main, label_str, (30,40), 0, 1, (255,0,0), 2, cv2.LINE_AA)

        label_str = '2. Original Image: '
        main = cv2.putText(main, label_str, (1290,20), 0, 0.5, (255,0,0), 2, cv2.LINE_AA)

        label_str = '3. Grayscaled Image: '
        main = cv2.putText(main, label_str, (1290,190), 0, 0.5, (255,0,0), 2, cv2.LINE_AA)

        label_str = '4. Thresholded Image: '
        main = cv2.putText(main, label_str, (1290,360), 0, 0.5, (255,0,0), 2, cv2.LINE_AA)

        label_str = '5. Warped Image: '
        main = cv2.putText(main, label_str, (1400,530), 0, 0.5, (255,0,0), 2, cv2.LINE_AA)

        cv2.imshow("frame", main)
        if cv2.waitKey(2) & 0xff == ord('q'):
            cv2.destroyAllWindows()
