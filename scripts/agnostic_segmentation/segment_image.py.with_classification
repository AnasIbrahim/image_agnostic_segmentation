#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse

import agnostic_segmentation
import compute_grasp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-image-path", type=str, help="path rgb to image", default='../../demo/rgb_images/000000.png')
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model",
                        default='../../models/FAT_trained_Ml2R_bin_fine_tuned.pth')
    parser.add_argument("--objects-images-folder", type=str,
                        help="path to object images to be segmented, ex: objects_folder/[book,shampoo,...]",
                        default='../../demo/objects')
    parser.add_argument("--compute_grasp", type=bool,
                        help="make the sample compute grasp suction points. Also depth images path is required then",
                        default=True)
    parser.add_argument("--depth-image-path", type=str, help="path to depth image", default='../../demo/depth_images/000000.png')
    parser.add_argument("--list", nargs="+", default=["a", "b"])
    parser.add_argument('-c-matrix', nargs='+',
                        help='camera matrix to convert depth image to point cloud',
                        default=['1390.53', '0.0', '964.957', '0.0', '1386.99', '522.586', '0.0', '0.0', '1.0'])
    args = parser.parse_args()

    # get absolute path
    args.rgb_image_path = os.path.abspath(args.rgb_image_path)
    args.model_path = os.path.abspath(args.model_path)
    args.objects_images_folder = os.path.abspath(args.objects_images_folder)
    if compute_grasp:
        args.depth_image_path = os.path.abspath(args.depth_image_path)
        depth_image = cv2.imread(args.depth_image_path, -1)
        depth_image = depth_image/1000  # convert from mm to meter - this is done due to depth images being saves in BOP format
        c_matrix = [float(i) for i in args.c_matrix]
        c_matrix = np.array(c_matrix).reshape((3, 3))

    rgb_img = cv2.imread(args.rgb_image_path)
    predictions = agnostic_segmentation.segment_image(rgb_img, args.model_path)
    seg_img = agnostic_segmentation.draw_segmented_image(rgb_img, predictions)

    cv2.imshow('segmented_image', seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # classify objects
    if args.objects_images_folder is not None:
        objects = os.listdir(args.objects_images_folder)
        for obj in objects:
            print('--------------------------------')
            print('Searching for object: ', obj)
            object_instances = agnostic_segmentation.find_object_mask(rgb_img, os.path.join(args.objects_images_folder,obj), predictions)
            if len(object_instances) == 0:
                print('No matches found for object: ', obj)
            else:
                print('Found ', str(len(object_instances)), ' matches for object ', obj)
                found_masks_img = agnostic_segmentation.draw_found_masks(rgb_img, object_instances, obj)

                cv2.imshow(obj + '  masks', found_masks_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # show suction points for all objects
    if compute_grasp:
        suction_pts = compute_grasp.compute_suction_points(rgb_img, depth_image, c_matrix, predictions)
        suction_pts_image = compute_grasp.visualize_suction_points(rgb_img, c_matrix, suction_pts)

        cv2.imshow('suction points', suction_pts_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
