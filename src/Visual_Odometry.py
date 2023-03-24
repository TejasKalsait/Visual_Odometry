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

first_img_left = None
first_img_flag = True
second_img_left = None

first_image_right = None
second_image_right = None
baseline = 0.07
focal_length = 476.7030

cx = 400
cy = 400


def visualodometry_callback(left_msg, right_msg):

    #print("Callback started")

    global first_img_left, second_img_left, first_img_flag, baseline, focal_length, first_img_right, second_image_right, cx, cy

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

    if first_img_flag:
        first_img_left = left_img
        first_img_right = right_img
        first_img_flag = False
        print("Returning")

        return
    else:
        # First image is taken
        second_img_left = left_img
        second_img_right = right_img


    orb = cv.ORB_create(nfeatures = 100, edgeThreshold = 20, fastThreshold = 20)

    temp_first_image_left_corners = orb.detect(first_img_left, None)

    #temp_img = cv.drawKeypoints(left_img, corners, None, color = (255, 0, 0))

    if len(temp_first_image_left_corners) == 0:
        return
    
    #print("How many corners", len(left_img_corners))

    # plt.imshow(temp_img)
    # plt.show()

    stereo = cv.StereoBM_create()
    
    stereo.setNumDisparities(64)
    stereo.setBlockSize(27)
    # stereo.setMinDisparity(0)
    stereo.setPreFilterType(1)
    stereo.setPreFilterSize(5)
    stereo.setPreFilterCap(21)

    # stereo.setSpeckleRange(32)
    # stereo.setSpeckleWindowSize(21)

    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(15)

    stereo.setDisp12MaxDiff(-1)

    second_disparity = stereo.compute(second_img_left, second_img_right)
    #print("Disparty length", second_disparity.shape)

    # This is the Depth Map
    
    second_depth = (focal_length * baseline) / second_disparity


    # print("Shape of Depth", depth.shape)
    # print("Values of Depth", depth[400, 100 : 150])

    
    # print("Type of a corner", type(first_img_corners[0]))

    # Find motion and corners in the Left+1 image

    first_image_left_corners = []
    
    for corner in temp_first_image_left_corners:
        first_image_left_corners.append(tuple([corner.pt[0], corner.pt[1]]))

    first_image_left_corners = np.array(first_image_left_corners).astype(np.float32)

    # temp_points = cv.goodFeaturesToTrack(first_img_left, maxCorners = 100)

    # print("Shape of First Image corners", temp_points.shape)
    # print("Few values", temp_points[0:5])
    # print("Type", type(temp_points))
    
    # print("Shape of First Image corners", first_image_left_corners.shape)
    # print("Few values", first_image_left_corners[0:5])

    cal_param = dict(winSize = (15, 15),
                     maxLevel = 2,
                     criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    second_image_left_corners, st, err = cv.calcOpticalFlowPyrLK(first_img_left, second_img_left, first_image_left_corners, None, **cal_param)

    #print("Length of next points", second_image_left_corners[5])

    three_d_pose = []

    temp = np.array([[1, 0, 0, -cx],
                             [0, 1, 0, -cy],
                             [0, 0, 0, -(focal_length)],
                             [0, 0, -(1/baseline), 0]], np.float32)
    
    reprojection = np.array([temp])

    # print(reprojection)
    # print(second_image_left_corners[5][0])
    # print(second_depth[int(second_image_left_corners[5][0])][int(second_image_left_corners[5][1])])
    
    for i in range(30):
        two_d_pose = np.array([second_image_left_corners[i][0], second_image_left_corners[i][1], second_depth[int(second_image_left_corners[i][0])][int(second_image_left_corners[i][1])], 1.0])
        three_d_pose.append(np.dot(reprojection, two_d_pose))
    
    print("ThreeDPose Shape", np.array(three_d_pose)[0])


    



if __name__ == "__main__":

    rospy.init_node("Visual_Odometry_Node")

    rate = rospy.Rate(10)

    left_sub = message_filters.Subscriber("/car_1/camera/left/image_raw/compressed", CompressedImage)
    right_sub = message_filters.Subscriber("/car_1/camera/right/image_raw/compressed", CompressedImage)
    odom_sub = message_filters.Subscriber("/car_1/base/odom", Odometry)

    depth_sub = message_filters.Subscriber("/car_1/camera/depth", String)

    ts = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub], queue_size = 10, slop = 1)
    print("TimeSync Done, Registerning callback")

    ts.registerCallback(visualodometry_callback)

    while not rospy.is_shutdown():
        rate.sleep()