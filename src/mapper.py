#! /usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid

if __name__ == "__main__":

    rospy.init_node("map_pub")

    pub = rospy.Publisher(name="/map", data_class=OccupancyGrid, queue_size=1)
    rate = rospy.Rate(1.0)

    map_data = [0]

    map = OccupancyGrid()

    map.info.height = 1
    map.info.width = 1
    # Resolution 250 because reference goes upto x = 200
    map.info.resolution = 400

    map.info.origin.position.x = -200
    map.info.origin.position.y = -200

    map.data = map_data


    # publish map here
    while not rospy.is_shutdown():
        pub.publish(map)
        rate.sleep()