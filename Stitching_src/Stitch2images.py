# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 04:52:33 2020
cv2.__version__ = 3.4.6.12
python 3.7
@author: HAOLEE
"""
import cv2
import os
import Stitching
from matplotlib import pyplot as plt

#your dataset path
path = 'D:\Downloads\images' #your dataset path
os.chdir(path) 
files = os.listdir(path)

#change the range to select images
for i in range(0,1):
    imgs = []

    dir = os.getcwd() + '\\' + files[i]
    print(dir)
    image = cv2.imread(dir)
    image = cv2.resize(image,(400 ,300))
    #image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    #image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    imgs.append(image)
    dir = os.getcwd() + '\\' + files[i+1]
    print(dir)
    image = cv2.imread(dir)
    image = cv2.resize(image,(400, 300))
    #image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    #image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    imgs.append(image)

    #cv2.imshow("result",image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    fileNameList = [(imgs[0],imgs[1])]

    for fname1, fname2 in fileNameList:
        # Read the img file
        #src_path = "img/"
        #fileName1 = fname1
        #fileName2 = fname2
        #img_left = cv2.imread(src_path + fileName1 + ".jpg")
        #img_right = cv2.imread(src_path + fileName2 + ".jpg")
        # The stitch object to stitch the image
        blending_mode = "linearBlending" # three mode - noBlending、linearBlending、linearBlendingWithConstant
        stitcher = Stitching.Stitcher()
        warp_img = stitcher.stitch([fname1, fname2], blending_mode)
        # plot the stitched image
        #warp_img = warp_img[:,:,::-1].astype(int)
        #cv2.imshow("result",warp_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        plt.figure(13)
        plt.title("warp_img")
        plt.imshow(warp_img[:,:,::-1].astype(int))

        # save the stitched iamge
        path = "./out_"+ str(i) +".jpg"
        saveFilePath = path.format(fname1, fname2, blending_mode)
        #warp_img= cv2.resize(warp_img,(1000, 400))
        cv2.imwrite(saveFilePath, warp_img)
