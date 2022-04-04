#!/usr/bin/env python

import rospy
import sys
import numpy as np
import os

import rospkg

from cv_bridge import CvBridge
bridge = CvBridge()

from image_agnostic_segmentation.srv import SegmentImage, SegmentImageResponse

rospack = rospkg.RosPack()
rospack.list()
lib_path = rospack.get_path('image_agnostic_segmentation')
lib_path = os.path.join(lib_path, '../scripts')
sys.path.insert(0, lib_path)
from agnostic_segmentation import agnostic_segmentation
from agnostic_segmentation import compute_grasp

def handle_segment_image(req):
    response = SegmentImageResponse()
    
    rgb_img = bridge.imgmsg_to_cv2(req.rgb_image, desired_encoding='passthrough')
    depth_img = bridge.imgmsg_to_cv2(req.depth_image, desired_encoding='passthrough')
    c_matrix = np.array(req.cam_K_matrix).reshape((3,3))

    # TODO replace with a ROS param
    model_path = '/home/gouda/segmentation/data_collection_ws/src/image_agnostic_segmentation/models/FAT_trained_Ml2R_bin_fine_tuned.pth'

    predictions = agnostic_segmentation.segment_image(rgb_img, model_path)
    seg_img = agnostic_segmentation.draw_segmented_image(rgb_img, predictions)

    suction_pts = compute_grasp.compute_suction_points(rgb_img, depth_img, c_matrix, predictions)
    suction_pts_image = compute_grasp.visualize_suction_points(seg_img, c_matrix, suction_pts)

    return response

def main():
    rospy.init_node('image_agnostic_segmentation')
    s = rospy.Service('segment_image', SegmentImage, handle_segment_image)
    rospy.loginfo("Segmentation service started.")
    rospy.spin()

if __name__ == "__main__":
    main()