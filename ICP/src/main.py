#!/usr/bin/env python

# -*- coding: utf-8 -*-
# How to use iterative closest point
# http://pointclouds.org/documentation/tutorials/iterative_closest_point.php#iterative-closest-point

import pcl
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2

# from pcl import icp, gicp, icp_nl


def main():
    pub_1 = rospy.Publisher('map', String, queue_size=500)
    pub_2 = rospy.Publisher('source', String, queue_size=500)
    cloud_in = load_XYZRGB('/home/jimmy/catkin_ws/src/EECS504_Final_Project/data/Map.ply')
    map_cloud = pc2.read_points(cloud_in)
    map_cloud.header.frame_id = 'world'

    count = 1
    while(count < 12):
        path_file = '/home/jimmy/catkin_ws/src/EECS504_Final_Project/data/Drone_path' + count + '.ply'
        cloud_out = load_XYZRGB( path_file)
        rospy.init_node('ICP Localization', anonymous=True)

        # std::cout << "Transformed " << cloud_in->points.size () << " data points:" << std::endl;
        print('Transformed ' + str(cloud_in.size) + ' data points:')

        # for (size_t i = 0; i < cloud_out->points.size (); ++i)
        #   std::cout << "    " << cloud_out->points[i].x << " " << cloud_out->points[i].y << " " << cloud_out->points[i].z << std::endl;
        for i in range(0, cloud_out.size):
            print('     ' + str(cloud_out[i][0]) + ' ' + str(cloud_out[i]
                                                             [1]) + ' ' + str(cloud_out[i][2]) + ' data points:')

        # pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        # icp.setInputCloud(cloud_in);
        # icp.setInputTarget(cloud_out);
        # pcl::PointCloud<pcl::PointXYZ> Final;
        # icp.align(Final);
        print('Start ICP.')
        icp = cloud_in.make_IterativeClosestPoint()
        # Final = icp.align()
        converged, transf, estimate, fitness = icp.icp(cloud_in, cloud_out)

        result[count-1] = pc2.read_points(estimate)
        result[count-1].header.frame_id = 'world'
        print(count)

        count += 1


        # std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
        # std::cout << icp.getFinalTransformation() << std::endl;
        # print('has converged:' + str(icp.hasConverged()) + ' score: ' + str(icp.getFitnessScore()) )
        # print(str(icp.getFinalTransformation()))
        print('has converged:' + str(converged) + ' score: ' + str(fitness))
        #print(str(transf))
        print('ICP is done.')
    
    rate = rospy.Rate(1) #1hz
    count = 0

    while not rospy.is_shutdown():
        
        pub_1.publish(map_cloud)
        pub_2.publish(result[count-1])
        print (count)
        if count < 13:
            count += 1
        else:
            count = 1 

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()