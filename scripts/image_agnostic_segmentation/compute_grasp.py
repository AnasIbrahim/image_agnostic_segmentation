import numpy as np
import open3d as o3d
import cv2
import copy

from vector_quaternion import pose_from_vector3D

def compute_suction_points(rgb_img, depth_img, c_matrix, predictions):
    # compute best suction point per mask
    instances = predictions["instances"].to("cpu")

    suction_pts = list()
    for i in range(len(instances)):
        pred_masks = instances.pred_masks[i].cpu().detach().numpy()

        # mask depth and rgb images
        masked_rgb_img = rgb_img.copy()
        masked_rgb_img = cv2.cvtColor(masked_rgb_img, cv2.COLOR_BGR2RGB)
        masked_rgb_img[pred_masks == False] = np.array([0, 0, 0])

        masked_depth_img = depth_img.copy()
        masked_depth_img = masked_depth_img.astype(float)
        masked_depth_img[pred_masks == False] = 0
        masked_depth_img = np.float32(masked_depth_img)

        #cv2.imshow('masked rgb image', masked_rgb_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #cv2.imshow('masked depth image', masked_depth_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # convert images to open3d types
        masked_rgb_img = o3d.geometry.Image(masked_rgb_img)
        masked_depth_img = o3d.geometry.Image(masked_depth_img)

        # convert image to point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                      c_matrix[0,0], c_matrix[1,1], c_matrix[0,2], c_matrix[1,2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(masked_rgb_img, masked_depth_img,
                                                                  depth_scale=1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        #o3d.visualization.draw_geometries([pcd])

        # segment plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.005, ransac_n=5, num_iterations=1000)
        plane_cloud = pcd.select_by_index(inliers)
        #plane_cloud.paint_uniform_color([1.0, 0, 0])


        plane_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        plane_cloud.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        #o = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        #o3d.visualization.draw_geometries([o, plane_cloud])

        grasp_position = plane_cloud.get_center()
        pcd_tree = o3d.geometry.KDTreeFlann(plane_cloud)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(grasp_position, 1)
        grasp_orientation = pose_from_vector3D(np.asarray(plane_cloud.normals)[idx[0], :])

        # visualize to check grasp computation
        grasp_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.015, cylinder_height=0.05, cone_height=0.02)
        grasp_arrow.paint_uniform_color(np.array([255,0,0]))
        grasp_arrow.translate(grasp_position)
        rot_mat = o3d.geometry.get_rotation_matrix_from_quaternion(grasp_orientation)
        #grasp_arrow.rotate(rot_mat)
        o = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d.visualization.draw_geometries([o, pcd, grasp_arrow])

        suction_pts.append((tuple(grasp_position), grasp_orientation))

    return suction_pts


def visualize_suction_points(rgb_img, c_matrix, suction_pts):
    suction_pts_img = copy.deepcopy(rgb_img)
    for point in suction_pts:
        # project 3D grasp point on image
        # mask rgb with the grasp point
        pass

    return suction_pts_img