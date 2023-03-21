#! /usr/bin/env python3

import rospy
import numpy as np
import cv2 as cv
from sensor_msgs.msg import CompressedImage, Image
import message_filters
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from std_msgs.msg import String
from nav_msgs.msg import Odometry

first_img = None
second_img = None


def visualodometry_callback(left_msg, right_msg, odom_msg, depth_msg):

    print("Callback started")

    bridge = CvBridge()
 
    odom_pub = rospy.Publisher("/car_1/odom", Odometry, queue_size = 10)

    # Converting compressed images to decompressed cv2 Gray Images

    temp_arr = np.fromstring(left_msg.data, np.uint8)
    left_img = cv.imdecode(temp_arr, flags = (cv.IMREAD_COLOR))
    left_img = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)

    # plt.imshow(left_img, cmap = "gray")
    # print("Showing the left image now")
    # plt.show()


    temp_arr = np.fromstring(right_msg.data, np.uint8)
    right_img = cv.imdecode(temp_arr, flags = (cv.IMREAD_COLOR))
    right_img = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)


    orb = cv.ORB_create(nfeatures = 100, edgeThreshold = 20, fastThreshold = 20)

    left_img_corners = orb.detect(left_img, None)
    right_img_corners = orb.detect(right_img, None)

    #temp_img = cv.drawKeypoints(left_img, corners, None, color = (255, 0, 0))

    if len(left_img_corners) == 0 or len(right_img_corners) == 0:
        return
    
    print("How many corners", len(left_img_corners), len(right_img_corners))

    # plt.imshow(temp_img)
    # plt.show()

    # Tracking Features using Optical Flow







if __name__ == "__main__":

    rospy.init_node("Visual_Odometry_Node")

    rate = rospy.Rate(10)

    left_sub = message_filters.Subscriber("/car_1/camera/left/image_raw/compressed", CompressedImage)
    right_sub = message_filters.Subscriber("/car_1/camera/right/image_raw/compressed", CompressedImage)
    odom_sub = message_filters.Subscriber("/car_1/base/odom", Odometry)

    depth_sub = message_filters.Subscriber("/car_1/camera/depth", String)

    ts = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub, odom_sub, depth_sub], queue_size = 10, slop = 1)
    print("TimeSync Done, Registerning callback")

    ts.registerCallback(visualodometry_callback)

    while not rospy.is_shutdown():
        rate.sleep()