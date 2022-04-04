#!/usr/bin/env python

import rospy
import sys
import numpy as np
import os

import rospkg
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
bridge = CvBridge()

from image_agnostic_segmentation.srv import SegmentImage, SegmentImageResponse
from image_agnostic_segmentation.msg import ImagePixels
from sensor_msgs.msg import Image

rospack = rospkg.RosPack()
rospack.list()
pkg_path = rospack.get_path('image_agnostic_segmentation')
lib_path = os.path.join(pkg_path, '../scripts')
sys.path.insert(0, lib_path)
from agnostic_segmentation import agnostic_segmentation
from agnostic_segmentation import compute_grasp

def handle_segment_image(req):
    rospy.loginfo("Segmentation service called.")

    seg_pub = rospy.Publisher('segmented_image', Image, queue_size=10)
    grasp_pub = rospy.Publisher('grasp_image', Image, queue_size=10)

    response = SegmentImageResponse()
    
    rgb_img = bridge.imgmsg_to_cv2(req.rgb_image, desired_encoding='passthrough')
    depth_img = bridge.imgmsg_to_cv2(req.depth_image, desired_encoding='passthrough')
    c_matrix = np.array(req.cam_K_matrix).reshape((3,3))

    # TODO replace with a ROS param
    model_path = os.path.join(pkg_path, '../models/FAT_trained_Ml2R_bin_fine_tuned.pth')

    predictions = agnostic_segmentation.segment_image(rgb_img, model_path)
    seg_img = agnostic_segmentation.draw_segmented_image(rgb_img, predictions)
    seg_img_msg = bridge.cv2_to_imgmsg(seg_img, encoding="rgb8")
    seg_pub.publish(seg_img_msg)
    rospy.loginfo("Published segmented image.")

    suction_pts = compute_grasp.compute_suction_points(rgb_img, depth_img, c_matrix, predictions)
    suction_pts_image = compute_grasp.visualize_suction_points(seg_img, c_matrix, suction_pts)
    suction_pts_image_msg = bridge.cv2_to_imgmsg(suction_pts_image, encoding="rgb8")
    grasp_pub.publish(suction_pts_image_msg)

    rospy.loginfo("Published grasp image.")

    # Service respose
    #masks_image = np.zeros_like(rgb_img)
    instances = predictions["instances"].to("cpu")
    objects_pixels = list()
    point_clouds = list()
    grasps = list()
    for i in range(len(instances)):
        mask = instances.pred_masks[i].cpu().detach().numpy()
        u, v = np.where(mask == True)
        object_pixels = ImagePixels()
        object_pixels.u_pixels = u
        object_pixels.v_pixels = v
        objects_pixels.append(object_pixels)
        #color = list(np.random.choice(range(256), size=3))
        #masks_image[mask == True] = color
        grasps.append(make_grasp_msg(suction_pts[i]))
    response.objects_pixels = np.array(objects_pixels)
    response.grasps = np.array(grasps)

    return response


def make_grasp_msg(pt):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'camera_optical_frame' # TODO change to ROS Param
    msg.pose.position.x = pt[0][0]
    msg.pose.position.y = pt[0][1]
    msg.pose.position.z = pt[0][2]
    msg.pose.orientation.x = pt[1][0]
    msg.pose.orientation.y = pt[1][1]
    msg.pose.orientation.z = pt[1][2]
    msg.pose.orientation.w = pt[1][3]

    return msg


def main():
    rospy.init_node('image_agnostic_segmentation')
    s = rospy.Service('segment_image', SegmentImage, handle_segment_image)
    rospy.loginfo("Segmentation service started.")
    rospy.spin()

if __name__ == "__main__":
    main()