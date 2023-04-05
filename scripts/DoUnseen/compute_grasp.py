import numpy as np
import open3d as o3d
import cv2
import copy

from .vector_quaternion import pose_from_vector3D

def make_predicted_objects_clouds(rgb_img, depth_img, c_matrix, predictions):
    instances = predictions["instances"].to("cpu")

    point_clouds = list()
    for i in range(len(instances)):
        pred_masks = instances.pred_masks[i].cpu().detach().numpy()

        # mask depth and rgb images
        masked_rgb_img = rgb_img.copy()
        masked_rgb_img = cv2.cvtColor(masked_rgb_img, cv2.COLOR_BGR2RGB)
        masked_rgb_img[pred_masks == False] = np.array([0, 0, 0])

        masked_depth_img = depth_img.copy()
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
                                                      c_matrix[0, 0], c_matrix[1, 1], c_matrix[0, 2], c_matrix[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(masked_rgb_img, masked_depth_img,
                                                                  depth_scale=1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        point_clouds.append(pcd)

    return point_clouds


def compute_suction_points(predictions, objects_point_clouds):
    # compute best suction point per mask
    instances = predictions["instances"].to("cpu")

    suction_pts = list()
    for i in range(len(instances)):
        pcd = objects_point_clouds[i]

        #o3d.visualization.draw_geometries([pcd])

        # segment plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.005, ransac_n=5, num_iterations=1000)
        plane_cloud = pcd.select_by_index(inliers)
        #plane_cloud.paint_uniform_color([1.0, 0, 0])


        plane_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(0.02))
        #plane_cloud.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        #plane_cloud.orient_normals_towards_camera_location(np.array([0,0,1]))
        #plane_cloud.orient_normals_consistent_tangent_plane(50)
        plane_cloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))

        #o = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        #o3d.visualization.draw_geometries([o, plane_cloud])

        grasp_position = plane_cloud.get_center()
        pcd_tree = o3d.geometry.KDTreeFlann(plane_cloud)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(grasp_position, 1)

        point = np.asarray(plane_cloud.points)[idx[0], :]
        normal = np.asarray(plane_cloud.normals)[idx[0], :]
        grasp_orientation = pose_from_vector3D(point-normal)

        # visualize to check grasp computation
        #grasp_arrow = make_grasp_arrow_cloud(grasp_position, grasp_orientation)
        #o = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        #o3d.visualization.draw_geometries([o, plane_cloud, grasp_arrow])

        suction_pts.append((tuple(grasp_position), tuple(grasp_orientation)))

    return suction_pts


def visualize_suction_points(rgb_img, c_matrix, suction_pts):
    suction_pts_img = copy.deepcopy(rgb_img)
    for point in suction_pts:
        grasp_position = np.array(point[0])
        grasp_orientation = np.array(point[1])

        # visualize to check grasp computation
        grasp_arrow = make_grasp_arrow_cloud(grasp_position, grasp_orientation)

        # project 3D grasp point on image
        arrow_cloud = grasp_arrow.sample_points_uniformly(number_of_points=50000)
        arrow_point = np.asarray(arrow_cloud.points)
        rvec = np.array([0, 0, 0], np.float)
        tvec = np.array([0, 0, 0], np.float)
        impoints, _ = cv2.projectPoints(arrow_point, rvec, tvec, c_matrix, None)
        impoints = np.squeeze(impoints, axis=1)
        impoints = impoints.astype(np.int)
        impoints = (impoints[:,1],impoints[:,0])

        # mask rgb with the grasp point
        try:
            suction_pts_img[impoints] = (0,255,0)
        except: # if pixel outside of image range (part of the arrow then), do nothing
            pass

    return suction_pts_img

def make_grasp_arrow_cloud(grasp_position, grasp_orientation):
    # TODO arrow is too small in new open3D release - make size 10X#
    grasp_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.0075,
                                                         cylinder_height=0.05, cone_height=0.02)
    grasp_arrow.paint_uniform_color(np.array([1, 0, 0]))
    grasp_arrow.translate((0, 0, -0.03))
    grasp_arrow.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0)))
    grasp_arrow.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi)))
    # grasp_arrow.translate((-0.03,0,0))
    rot_mat = o3d.geometry.get_rotation_matrix_from_quaternion(grasp_orientation)
    grasp_arrow.rotate(rot_mat)
    grasp_arrow.translate(grasp_position)
    return grasp_arrow
