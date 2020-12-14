# ICP-Localization-by-SfM-and-image-stitching
This is the public repo for the final project of EECS 504

The goal of the project is to construct the 3D map from 2D images, and then use the local 2D image information do the localization tracking. The main localization algorithm is Iterative closest point (ICP).

A series of drone images is used as the example. The results were recorded as a video.

Dependencies:
	Ubuntu 18.04
	ROS melodic
	Python > 3.5.x
	pcl 1.8.1	https://github.com/strawlab/python-pcl
	Cython <= 0.25.2

Dataset:
OpenDrone Map Dataset (ODM Dataset)		https://drive.google.com/drive/folders/1O1TIR0ohgkf4xtJx7RsKn5us14D-L_xB