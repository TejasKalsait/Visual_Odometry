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
# focal_length = None

cx = 400
cy = 400

def visualodometry_callback(left_msg, right_msg):

    # Defining Global variables
    global first_img_left, second_img_left, first_img_flag, baseline, focal_length, first_img_right, second_image_right, cx, cy
    
    print("Callback started")
    print("Focal_length is", focal_length)

    # Constructor to convert image < > cv2
    bridge = CvBridge()

    # Odom Pulisher
    odom_pub = rospy.Publisher("/car_1/odom", Odometry, queue_size = 10)

    # Getting the left image
    left_img = bridge.compressed_imgmsg_to_cv2(left_msg, "bgr8")
    left_img = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)

    # Getting the second image
    right_img = bridge.compressed_imgmsg_to_cv2(right_msg, "bgr8")
    right_img = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

    # Storing the first image
    if first_img_flag:
        first_img_left = left_img
        first_img_right = right_img
        first_img_flag = False
        print("Returning because first frame")

        return
    else:
        # First image is already taken
        second_img_left = left_img
        second_img_right = right_img

    # WE HAVE first_img_left/right and second_img_left/right

    # ORB detector constructor
    orb = cv.ORB_create(nfeatures = 100, edgeThreshold = 20, fastThreshold = 20)

    # Detecting corners
    temp_first_image_left_corners = orb.detect(first_img_left, None)

    # Return if no corners
    if len(temp_first_image_left_corners) == 0:
        print("Returning because algorithm couldn't find corners")
        return

    # To Visualize the corners
    #first_img_left_draw = cv.drawKeypoints(first_img_left, temp_first_image_left_corners, None, color = (255, 0, 0))

    # plt.imshow(first_img_left_draw)
    # print("Showing Corners in first image now and ", len(temp_first_image_left_corners), "Corners detected")
    # plt.show()

    # Constructor for Disparity map
    stereo = cv.StereoBM_create()

    # Parameters to change
    stereo.setNumDisparities(32)
    stereo.setBlockSize(31)
    # stereo.setTextureThreshold(10)
    # stereo.setUniquenessRatio(15)
    stereo.setDisp12MaxDiff(-1)
    # stereo.setPreFilterType(1)
    # stereo.setPreFilterSize(5)
    # stereo.setPreFilterCap(21)
    stereo.setMinDisparity(0)
    # stereo.setSpeckleRange(32)
    # stereo.setSpeckleWindowSize(21)

    # Creating the Disparity Map
    second_disparity = stereo.compute(second_img_left, second_img_right)
    second_disparity = np.clip(second_disparity, 0.0, float('inf'))

    # Displaying the Disparity Map
    plt.imshow(second_disparity,cmap = 'gray')
    print("Showing Disparity of shape", second_disparity.shape)
    print("Max and Min of disparity", second_disparity.max(), second_disparity.min())
    plt.show()

    # Preprocess for optical flow
    first_image_left_corners = []

    for corner in temp_first_image_left_corners:
        first_image_left_corners.append(tuple([corner.pt[0], corner.pt[1]]))

    first_image_left_corners = np.array(first_image_left_corners).astype(np.float32)

    # Parametes for Optical flow
    cal_param = dict(winSize = (15, 15),
                     maxLevel = 2,
                     criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Computing optical flow
    second_image_left_corners, st, err = cv.calcOpticalFlowPyrLK(first_img_left, second_img_left, first_image_left_corners, None, **cal_param)

    # Calculating the 3D point cloud
    three_d_poses = []
    num_corners_to_track = 30

    reprojection = np.array([[1, 0, 0, -cx],
                             [0, 1, 0, -cy],
                             [0, 0, 0, -(focal_length)],
                             [0, 0, -(1/baseline), 0]], np.float32)
    print("Shape of reprojection", reprojection.shape)

    #second_depth = np.clip(second_depth, 0.0, float('inf'))
    
    for i in range(num_corners_to_track):
        second_corner_x, second_corner_y = second_image_left_corners[i][0], second_image_left_corners[i][1]
        #second_depth = np.clip(second_depth, 0.0, float('inf'))
        second_corner_depth = second_disparity[int(second_corner_x)][int(second_corner_y)]

        two_d_pose = np.array([second_corner_x, second_corner_y, second_corner_depth, 1.0], np.float32)
        #second_depth = np.clip(second_depth, 0.0, float('inf'))
        three_d_pose = (reprojection @ two_d_pose)
        last_val = three_d_pose[3]
        #last_val = 1.0

        # if three_d_pose[0] / last_val == np.inf or three_d_pose[0] / last_val == -np.inf or three_d_pose[0] / last_val == np.nan:
        #     continue

        three_d_poses.append([three_d_pose[0]/ last_val, three_d_pose[1] / last_val, three_d_pose[2] / last_val, three_d_pose[3] / last_val])

    three_d_poses = np.array(three_d_poses)
    
    print("ThreeDPoses", three_d_poses)
    print("3D Poses Shape", three_d_poses.shape)

    first_img_left = second_img_left
    first_img_right = second_image_right

def focal_callback(focal_msg):

    global focal_length
    
    focal_length = (eval(str(focal_msg.data))[0] + eval(str(focal_msg.data))[4]) / 2

    # rospy.loginfo(focal_length)

    print("Focal length is", focal_length)
    print("Type is", type(focal_length))

def depth_callback(depth_msg):

    global second_depth

    bridge = CvBridge()

    second_depth = bridge.imgmsg_to_cv2(depth_msg)
    second_depth = np.clip(second_depth, 0.0, float('inf'))
    #print("Got Depth Shape as", second_depth.shape)






if __name__ == "__main__":

    rospy.init_node("Visual_Odometry_Node")

    rate = rospy.Rate(5)

    # focal = rospy.Subscriber("/car_1/camera/intrinsic", String, focal_callback, queue_size = 2)
    # depth = rospy.Subscriber("/car_1/camera/depth", Image, depth_callback, queue_size = 1)

    # rospy.sleep(3.0)

    left_sub = message_filters.Subscriber("/car_1/camera/left/image_raw/compressed", CompressedImage)
    right_sub = message_filters.Subscriber("/car_1/camera/right/image_raw/compressed", CompressedImage)

    #depth_sub = message_filters.Subscriber("/car_1/camera/depth", String)

    ts = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub], queue_size = 10, slop = 1)
    print("TimeSync Done, Registerning callback")

    ts.registerCallback(visualodometry_callback)

    while not rospy.is_shutdown():
        rate.sleep()