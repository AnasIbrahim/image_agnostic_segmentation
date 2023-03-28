#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse

from agnostic_segmentation import agnostic_segmentation
from agnostic_segmentation import compute_grasp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-image-path", type=str, help="path rgb to image", default='../demo/rgb_images/000000.png')
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model",
                        default='../models/FAT_trained_Ml2R_bin_fine_tuned.pth')
    parser.add_argument('--compute-suction-pts', dest='compute_suction_pts', action='store_true')
    parser.add_argument('--compute-no-suction-pts', dest='compute_suction_pts', action='store_false')
    parser.set_defaults(compute_suction_pts=True)
    parser.add_argument("--depth-image-path", type=str, help="path to depth image", default='../demo/depth_images/000000.png')
    parser.add_argument("--depth-scale", type=int, help="depth image in divided by the scale", default=1000)  # convert from mm to meter - this is done due to depth images being saves in BOP format
    parser.add_argument('-c-matrix', nargs='+',
                        help='camera matrix to convert depth image to point cloud',
                        default=['1390.53', '0.0', '964.957', '0.0', '1386.99', '522.586', '0.0', '0.0', '1.0']) # HOPE dataset - example images
    parser.add_argument('--detect-all-objects', dest='detect_all_objects', action='store_true')
    parser.set_defaults(detect_all_objects=False)
    parser.add_argument('--detect-one-object', type=str, help='name of object (folder) to be detected', default='obj_000016')
    parser.add_argument('--gallery_path', type=str, help='path to gallery images folder', default='../demo/objects_gallery')
    args = parser.parse_args()

    # get absolute path
    args.rgb_image_path = os.path.abspath(args.rgb_image_path)
    args.model_path = os.path.abspath(args.model_path)
    if args.compute_suction_pts:
        args.depth_image_path = os.path.abspath(args.depth_image_path)
        depth_image = cv2.imread(args.depth_image_path, -1)
        depth_image = depth_image/args.depth_scale
        c_matrix = [float(i) for i in args.c_matrix]
        c_matrix = np.array(c_matrix).reshape((3, 3))

    rgb_img = cv2.imread(args.rgb_image_path)
    predictions = agnostic_segmentation.segment_image(rgb_img, args.model_path)
    seg_img = agnostic_segmentation.draw_segmented_image(rgb_img, predictions)

    cv2.imshow('segmented_image', seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # show suction points for all objects
    if args.compute_suction_pts:
        objects_point_clouds = compute_grasp.make_predicted_objects_clouds(rgb_img, depth_image, c_matrix, predictions)
        suction_pts = compute_grasp.compute_suction_points(predictions, objects_point_clouds)
        suction_pts_image = compute_grasp.visualize_suction_points(seg_img, c_matrix, suction_pts)

        cv2.imshow('suction points', suction_pts_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.detect_all_objects:
        zero_shot_classifier = agnostic_segmentation.DoUnseen(args.gallery_path, method='vit')
        predictions = zero_shot_classifier.classify_all_objects(rgb_img, predictions)
    elif args.detect_one_object:
        zero_shot_classifier = agnostic_segmentation.DoUnseen(args.gallery_path, method='vit')
        predictions = zero_shot_classifier.find_object(rgb_img, predictions)



if __name__ == '__main__':
    main()
