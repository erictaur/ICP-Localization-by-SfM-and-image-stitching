# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 04:52:33 2020

@author: kenneth
"""
import cv2
import os
##import Stitching
#from matplotlib import pyplot as plt
path = 'D:\Downloads\images'
os.chdir(path) 
files = os.listdir(path)


#%%
def multi_imgs_stitching(file_index, direction, out_file):
    image_per_frame = []
    start, end = file_index
    if(direction==1):
        direction = cv2.cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        direction = cv2.cv2.ROTATE_90_CLOCKWISE
    for i in range(start,end):
        dir = os.getcwd() + '\\' + files[i]
        print(dir)
        image = cv2.imread(dir)
        image = cv2.resize(image,(400, 300))
        image = cv2.rotate(image, direction)
        image_per_frame.append(image)
        #cv2.imshow('ratate', image) 
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    stitcher = cv2.Stitcher_create()
    status, output_file = stitcher.stitch(image_per_frame)
    print("status = ", status)
    if status == 0:
        print("after stitch ")
        #output file path
        out_path = "./"+ out_file + ".JPG"
        cv2.imwrite(out_path, output_file) 
        cv2.imshow("Stitch", output_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return output_file

#f0t14 = multi_imgs_stitching((0,15),1,"S1")
##f15t22 = multi_imgs_stitching((15,23),0)
#f23t25 = multi_imgs_stitching((23,27),1,"test1")
#f26t27 = multi_imgs_stitching((26,29),1,"test2")
#f28t33 = multi_imgs_stitching((28,34),1,"test3")

multi_imgs_stitching((0,15),1,"S3")