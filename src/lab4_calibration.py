#! /usr/bin/env python3

import rospy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String

rospy.init_node("calibration_node")

intrinsic_pub = rospy.Publisher("/car_1/camera/intrinsic", String, queue_size = 2)


num_of_images = rospy.get_param("calib_image_num")

# Distance between neighbouring blobs
dist = 100

world_point = np.zeros((44, 3), np.float32)

world_space = []
image_space = []


first_block = True
x_delta = 0

for i in range(len(world_point)):
    
    if first_block:
        world_point[i][0] = x_delta
        world_point[i][1] = dist * (i % 4)
        if i % 4 == 3:
            first_block = False
            x_delta += dist // 2


    else:
        world_point[i][0] = x_delta
        world_point[i][1] = (dist * (i % 4)) + dist // 2
        if i % 4 == 3:
            first_block = True
            x_delta += dist // 2

    

#print(world_point[42])

def calibration():
    
    global num_of_images, dist, world_point, intrinsic_pub

    count = 0

    while count < num_of_images:

        rand_num = np.random.randint(low = 0, high = 469)
        #print("Random number is", rand_num)

        if rand_num < 10 and rand_num >= 0:
            img_read = "/home/cse4568/catkin_ws/src/lab4/calib/frame00000" + str(rand_num) + ".png"
        elif rand_num < 100 and rand_num > 9:
            img_read = "/home/cse4568/catkin_ws/src/lab4/calib/frame0000" + str(rand_num) + ".png"
        else:
            img_read = "/home/cse4568/catkin_ws/src/lab4/calib/frame000" + str(rand_num) + ".png"

        #print(img_read)

        img = cv.imread(img_read, cv.IMREAD_GRAYSCALE)

        # plt.imshow(img, cmap = 'gray')
        # plt.show()
        
        params = cv.SimpleBlobDetector_Params()

        params.minThreshold = 8
        params.maxThreshold = 255

        params.filterByInertia = True
        params.minInertiaRatio = 0.1

        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 2500

        params.filterByConvexity = True
        params.minConvexity = 0.80

        
        detector = cv.SimpleBlobDetector_create(params)

        blobs = detector.detect(img)
        #print(len(blobs))

        # Drawing keypoints
        blob_img = cv.drawKeypoints(img, blobs, np.array([]), (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # MAke Gray for Corner Sub Pix
        blob_img_gray = cv.cvtColor(blob_img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findCirclesGrid(blob_img_gray, (4, 11), None, flags = (cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING))

        # print(ret)

        # if ret == False:
        #     print("Image is", rand_num)

        #     plt.imshow(blob_img)
        #     plt.show()
        #     continue

        # print("How many corners", len(corners))

        if ret == True:

            world_space.append(world_point)

            corners = cv.cornerSubPix(blob_img_gray, corners, (5, 5), (-1, -1), criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            image_space.append(corners)

        count += 1

    #print("Calibrating...")
    ret, matrix, distortion, rotation, translation = cv.calibrateCamera(world_space, image_space, img.shape[: : -1], None, None)

    #print("RET", ret)
    #print("Rotation", rotation)
    #print("Translation", translation)
    #print("MTX", matrix)
    #print("Distortion", distortion)

    result = str(np.array(matrix).reshape(1, -1)[0])
    print(result)
    intrinsic_pub.publish(result)




if __name__ == '__main__':
    
    calibration()
    rospy.spin()
