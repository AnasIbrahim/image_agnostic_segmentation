#!/usr/bin/env python

import sys
import os

import cv2
import rospy

import rospkg
from cv_bridge import CvBridge
bridge = CvBridge()

from image_agnostic_segmentation.srv import SegmentImage, SegmentImageRequest

rospack = rospkg.RosPack()
rospack.list()
pkg_path = rospack.get_path('image_agnostic_segmentation')

def main():
    rospy.init_node('image_agnostic_segmentation_test')

    req = SegmentImageRequest()
    rgb_img = os.path.join(pkg_path, '../demo/rgb_images/000000.png')
    req.rgb_image = bridge.cv2_to_imgmsg(cv2.imread(rgb_img))
    depth_img = os.path.join(pkg_path, '../demo/depth_images/000000.png')
    req.depth_image = bridge.cv2_to_imgmsg(cv2.imread(depth_img))
    req.cam_K_matrix = [1390.53, 0.0, 964.957, 0.0, 1386.99, 522.586, 0.0, 0.0, 1.0]
    
    rospy.wait_for_service('segment_image')
    try:
        segment_image = rospy.ServiceProxy('segment_image', SegmentImage)
        resp = segment_image(req)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    main()