# ICP-Localization-by-SfM-and-image-stitching

### This is the public repo for the final project of EECS 504

#### EECS 504: Foundations of Computer Vision</br>University of Michigan, Ann Arbor

#### Collaborators:
Wei-Chin Chien
Ke-Haur Taur
Wei-Fan Tseng
Hao-Tsung Lee

### Main Objective

The goal of the project is to construct the 3D map from 2D images, and then use the local 2D image information do the localization tracking. The main localization algorithm is Iterative closest point (ICP).

### Editing/Viewing Point Clouds
Point Clouds are generated from the public OpenSfM package.
Software like Meshlab can view/edit .plt files which you can find in our /data folder.

The following figure illustrates our target map for localization.
![](https://i.imgur.com/AinbRTX.jpg)


### Performing Localization

A ROS envrionment with RViz installed is required for executing our ICP algorithm.

The figure below shows the third time step of our localization process. </br>Localized source labeled as the dark region in the figure.

![](https://i.imgur.com/98yAOl1.jpg)


A series of drone images are used as the our dataset. The results of our implementation were recorded as a video (See the video folder).

---

### Dependencies:
+ Ubuntu 18.04
+ ROS melodic
+ Python > 3.5.x
+ pcl 1.8.1	https://github.com/strawlab/python-pcl
+ Cython <= 0.25.2

### Dataset:
OpenDrone Map Dataset (ODM Dataset)		https://drive.google.com/drive/folders/1O1TIR0ohgkf4xtJx7RsKn5us14D-L_xB
