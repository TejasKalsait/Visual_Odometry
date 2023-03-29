#! /usr/bin/env python3

import rospy
import numpy as np
import cv2 as cv
from sensor_msgs.msg import CompressedImage, Image
import message_filters
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from std_msgs.msg import String

baseline = 0.07
# focal_length = None
focal_length = 476.7030

def depth_callback(left_msg, right_msg):

    global baseline, focal_length

    print("Callback started")

    assert left_msg.header.stamp == right_msg.header.stamp

    bridge = CvBridge()
    #dtype1, channels1 = bridge.encoding_as_cvtype2("bgr8")
    depth_pub = rospy.Publisher("/car_1/camera/depth", Image, queue_size = 10)

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


    # plt.imshow(right_img)
    # plt.imshow("Showing the right image now")
    # plt.show()


    # Starting the Depth Estimation Logic

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

    #stereo.setDisp12MaxDiff(-1)


    disparity = stereo.compute(left_img, right_img)
    #print("Disparty is", disparity)

    depth = ((focal_length * baseline) / disparity).astype(np.uint8)

    #print("Type is", type(depth))

    #depth = (depth - depth.min()) / (depth.max() - depth.min())

    #depth = (depth - depth[np.unravel_index(np.argmin(depth, axis = None), depth.shape)]) / (depth[np.unravel_index(np.argmax(depth, axis = None), depth.shape)] - depth[np.unravel_index(np.argmin(depth, axis = None), depth.shape)])

    #depth = np.uint8(depth)

    #print("Max is", depth.max(), "Min is", depth.min())

    # plt.imshow(depth, cmap = 'gray')
    # print("Displaying he depth image now")
    # plt.show()

    depth_pub.publish(bridge.cv2_to_imgmsg(depth, encoding = "mono8"))
    
    #depth = depth.astype(np.uint8)
    # depth_pub.publish(depth)


def focal_callback(focal_msg):

    global focal_length

    # stripped = str(focal_msg.data).strip(" []")
    # focal_length = float(stripped[0:12])
    
    focal_length = eval(str(focal_msg.data))[0]

    # rospy.loginfo(focal_length)

    print("Focal length is", focal_length)
    print("Type is", type(focal_length))


if __name__ == "__main__":

    rospy.init_node("Depth_Map_Node", anonymous = True)

    rate = rospy.Rate(12)

    # focal = rospy.Subscriber("/car_1/camera/intrinsic", String, focal_callback, queue_size = 2)


    # rospy.sleep(3.0)
    
    left_sub = message_filters.Subscriber("/car_1/camera/left/image_raw/compressed", CompressedImage)
    right_sub = message_filters.Subscriber("/car_1/camera/right/image_raw/compressed", CompressedImage)

    ts = message_filters.TimeSynchronizer([left_sub, right_sub], queue_size = 10)
    print("TimeSync Done, Registerning callback")

    ts.registerCallback(depth_callback)

    while not rospy.is_shutdown():
        rate.sleep()
