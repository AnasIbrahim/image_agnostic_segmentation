#!/usr/bin/env python
import copy
import sys
import os
import numpy as np
import cv2
import open3d as o3d

import rospy
import rospkg
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge
bridge = CvBridge()

from image_agnostic_segmentation.srv import SegmentImage, SegmentImageRequest

rospack = rospkg.RosPack()
rospack.list()
pkg_path = rospack.get_path('image_agnostic_segmentation')

def main():
    rospy.init_node('image_agnostic_segmentation_test')

    cloud_pub = rospy.Publisher('scene_point_cloud', PointCloud2, queue_size=10)

    req = SegmentImageRequest()
    rgb_img_path = os.path.join(pkg_path, '../demo/rgb_images/000000.png')
    req.rgb_image = bridge.cv2_to_imgmsg(cv2.imread(rgb_img_path))
    depth_img_path = os.path.join(pkg_path, '../demo/depth_images/000000.png')
    req.depth_image = bridge.cv2_to_imgmsg(cv2.imread(depth_img_path,-1)/1000)
    req.cam_K_matrix = [1390.53, 0.0, 964.957, 0.0, 1386.99, 522.586, 0.0, 0.0, 1.0]
    c_matrix = copy.deepcopy(np.array(req.cam_K_matrix)).reshape((3,3))

    # make and publish point cloud
    rgb_img = o3d.geometry.Image(cv2.cvtColor(cv2.imread(rgb_img_path), cv2.COLOR_BGR2RGB))
    depth_img = cv2.imread(depth_img_path, -1)/1000
    depth_img = o3d.geometry.Image(depth_img.astype(np.float32))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(np.asarray(rgb_img).shape[0], np.asarray(rgb_img).shape[1],
                                                  c_matrix[0, 0], c_matrix[1, 1], c_matrix[0, 2], c_matrix[1, 2])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img,
                                                              depth_scale=1, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    #o3d.visualization.draw_geometries([pcd])

    #pc_msg = o3d_to_pc_msg(pcd)
    pc_msg = o3dpc_to_rospc(pcd, stamp=rospy.Time.now(), frame_id='camera_optical_frame')

    cloud_pub.publish(pc_msg)

    rospy.wait_for_service('segment_image')
    try:
        segment_image = rospy.ServiceProxy('segment_image', SegmentImage)
        resp = segment_image(req)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


#def o3d_to_pc_msg(o3d_cloud):
#    npoints = len(np.asarray(o3d_cloud.points))
#    points_arr = np.zeros((npoints,), dtype=[
#        ('x', np.float32),
#        ('y', np.float32),
#        ('z', np.float32),
#        ('rgb', np.float32)])
#    points_color = np.zeros((npoints,), dtype=[
#        ('r', np.uint8),
#        ('g', np.uint8),
#        ('b', np.uint8)])
#    point_xyz = np.asarray(o3d_cloud.points)
#    points_arr['x'] = point_xyz[:,0]
#    points_arr['y'] = point_xyz[:,1]
#    points_arr['z'] = point_xyz[:,2]
#    points_rgb = np.asarray(o3d_cloud.colors)*255
#    points_color['r'] = points_rgb[:,0]
#    points_color['g'] = points_rgb[:,1]
#    points_color['b'] = points_rgb[:,2]
#    points_arr['rgb'] = ros_numpy.point_cloud2.merge_rgb_fields(points_color)
#
#    #cloud_msg = ros_numpy.msgify(PointCloud2, points_arr)
#    cloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(points_arr, stamp=rospy.Time.now(), frame_id='camera_optical_frame')
#
#    return cloud_msg


# copied from https://github.com/SeungBack/open3d-ros-helper/blob/master/open3d_ros_helper/open3d_ros_helper.py
def o3dpc_to_rospc(o3dpc, frame_id=None, stamp=None):
    """ convert open3d point cloud to ros point cloud
    Args:
        o3dpc (open3d.geometry.PointCloud): open3d point cloud
        frame_id (string): frame id of ros point cloud header
        stamp (rospy.Time): time stamp of ros point cloud header
    Returns:
        rospc (sensor.msg.PointCloud2): ros point cloud message
    """
    BIT_MOVE_16 = 2 ** 16
    BIT_MOVE_8 = 2 ** 8

    cloud_npy = np.asarray(copy.deepcopy(o3dpc.points))
    is_color = o3dpc.colors

    n_points = len(cloud_npy[:, 0])
    if is_color:
        data = np.zeros(n_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
    else:
        data = np.zeros(n_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ])
    data['x'] = cloud_npy[:, 0]
    data['y'] = cloud_npy[:, 1]
    data['z'] = cloud_npy[:, 2]

    if is_color:
        rgb_npy = np.asarray(copy.deepcopy(o3dpc.colors))
        rgb_npy = np.floor(rgb_npy * 255)  # nx3 matrix
        rgb_npy = rgb_npy[:, 0] * BIT_MOVE_16 + rgb_npy[:, 1] * BIT_MOVE_8 + rgb_npy[:, 2]
        rgb_npy = rgb_npy.astype(np.uint32)
        data['rgb'] = rgb_npy

    rospc = ros_numpy.msgify(PointCloud2, data)
    if frame_id is not None:
        rospc.header.frame_id = frame_id

    if stamp is None:
        rospc.header.stamp = rospy.Time.now()
    else:
        rospc.header.stamp = stamp
    rospc.height = 1
    rospc.width = n_points
    rospc.fields = []
    rospc.fields.append(PointField(
        name="x",
        offset=0,
        datatype=PointField.FLOAT32, count=1))
    rospc.fields.append(PointField(
        name="y",
        offset=4,
        datatype=PointField.FLOAT32, count=1))
    rospc.fields.append(PointField(
        name="z",
        offset=8,
        datatype=PointField.FLOAT32, count=1))

    if is_color:
        rospc.fields.append(PointField(
            name="rgb",
            offset=12,
            datatype=PointField.UINT32, count=1))
        rospc.point_step = 16
    else:
        rospc.point_step = 12

    rospc.is_bigendian = False
    rospc.row_step = rospc.point_step * n_points
    rospc.is_dense = True
    return rospc


if __name__ == "__main__":
    main()