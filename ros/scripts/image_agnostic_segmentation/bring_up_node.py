#!/usr/bin/env python

import rospy
import sys
import numpy as np
import os
import open3d as o3d

import rospkg
import ros_numpy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose

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

    seg_img_pub = rospy.Publisher('segmented_image', Image, queue_size=10)
    grasp_img_pub = rospy.Publisher('grasp_image', Image, queue_size=10)

    grasp_poses_pub = rospy.Publisher('grasp_poses', PoseArray, queue_size=10)

    response = SegmentImageResponse()
    
    rgb_img = bridge.imgmsg_to_cv2(req.rgb_image, desired_encoding='passthrough')
    depth_img = bridge.imgmsg_to_cv2(req.depth_image, desired_encoding='passthrough')
    c_matrix = np.array(req.cam_K_matrix).reshape((3,3))

    # TODO replace with a ROS param
    model_path = os.path.join(pkg_path, '../models/FAT_trained_Ml2R_bin_fine_tuned.pth')

    predictions = agnostic_segmentation.segment_image(rgb_img, model_path)
    seg_img = agnostic_segmentation.draw_segmented_image(rgb_img, predictions)
    seg_img_msg = bridge.cv2_to_imgmsg(seg_img, encoding="rgb8")
    seg_img_pub.publish(seg_img_msg)
    rospy.loginfo("Published segmented image.")

    objects_clouds = compute_grasp.make_predicted_objects_clouds(rgb_img, depth_img, c_matrix, predictions)
    suction_pts = compute_grasp.compute_suction_points(predictions, objects_clouds)
    suction_pts_image = compute_grasp.visualize_suction_points(seg_img, c_matrix, suction_pts)
    suction_pts_image_msg = bridge.cv2_to_imgmsg(suction_pts_image, encoding="rgb8")
    grasp_img_pub.publish(suction_pts_image_msg)

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
        point_clouds.append(o3d_to_pc_msg(objects_clouds[i]))
        grasps.append(suction_pts[i])

    grasp_poses_pub.publish(make_grasp_msg(grasps))

    response.objects_pixels = np.array(objects_pixels)
    response.objects_clouds = np.array(point_clouds)
    response.grasps = make_grasp_msg(grasps)

    rospy.loginfo("Service finished.")
    return response


def make_grasp_msg(grasps):
    msg = PoseArray()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'camera_optical_frame' # TODO change to ROS Param
    grasp_msgs = list()
    for grasp in grasps:
        pt = Pose()
        pt.position.x = grasp[0][0]
        pt.position.y = grasp[0][1]
        pt.position.z = grasp[0][2]
        pt.orientation.x = grasp[1][0]
        pt.orientation.y = grasp[1][1]
        pt.orientation.z = grasp[1][2]
        pt.orientation.w = grasp[1][3]
        grasp_msgs.append(pt)

    msg.poses = np.array(grasp_msgs)

    return msg

def o3d_to_pc_msg(o3d_cloud):
    npoints = len(np.asarray(o3d_cloud.points))
    points_arr = np.zeros((npoints,), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('r', np.uint8),
        ('g', np.uint8),
        ('b', np.uint8)])
    point_xyz = np.asarray(o3d_cloud.points)
    points_arr['x'] = point_xyz[:,0]
    points_arr['y'] = point_xyz[:,1]
    points_arr['z'] = point_xyz[:,2]
    points_rgb = np.asarray(o3d_cloud.colors)
    points_arr['r'] = points_rgb[:,0]
    points_arr['g'] = points_rgb[:,1]
    points_arr['b'] = points_rgb[:,2]

    cloud_msg = ros_numpy.msgify(PointCloud2, points_arr)

    return cloud_msg


def main():
    rospy.init_node('image_agnostic_segmentation')
    s = rospy.Service('segment_image', SegmentImage, handle_segment_image)
    rospy.loginfo("Segmentation service started.")
    rospy.spin()

if __name__ == "__main__":
    main()