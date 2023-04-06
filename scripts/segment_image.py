#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse

from DoUnseen.Dounseen import ZeroShotClassification, UnseenSegment, draw_segmented_image
from DoUnseen import compute_grasp

import torch
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-image-path", type=str, help="path rgb to image", default='../demo/rgb_images/000000.png')
    parser.add_argument("--segmentation-model-path", type=str, help="path to unseen object segmentation model",
                        default='../models/unseen_object_segmentation.pth')

    parser.add_argument('--classification-method', type=str, help="method to use for classification 'vit' or 'siamese'", default='vit')
    parser.add_argument("--siamese-model-path", type=str, help="path to siamese classification model",
                        default='../models/classification_siamese_net.pth')
    parser.add_argument('--detect-all-objects', dest='detect_all_objects', action='store_true')
    parser.set_defaults(detect_all_objects=False)
    parser.add_argument('--detect-one-object', dest='detect_one_object', action='store_true')
    parser.set_defaults(detect_one_object=False)
    parser.add_argument('--object-name', type=str, help='name of object (folder) to be detected', default='obj_000016')
    parser.add_argument('--gallery_images_path', type=str, help='path to gallery images folder', default='../demo/objects_gallery')
    parser.add_argument('--use-buffered-gallery', dest='use-buffered-gallery', action='store_true')
    parser.set_defaults(use_buffered_gallery=False)
    parser.add_argument('--gallery_buffered_path', type=str, help='path to buffered gallery file', default='../demo/objects_gallery_vit.pkl')

    parser.add_argument('--compute-suction-pts', dest='compute_suction_pts', action='store_true')
    parser.set_defaults(compute_suction_pts=False)
    parser.add_argument("--depth-image-path", type=str, help="path to depth image", default='../demo/depth_images/000000.png')
    parser.add_argument("--depth-scale", type=int, help="depth image in divided by the scale", default=1000)  # convert from mm to meter - this is done due to depth images being saves in BOP format
    parser.add_argument('-c-matrix', nargs='+',
                        help='camera matrix to convert depth image to point cloud',
                        default=['1390.53', '0.0', '964.957', '0.0', '1386.99', '522.586', '0.0', '0.0', '1.0']) # HOPE dataset - example images

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()

    # get absolute path
    args.rgb_image_path = os.path.abspath(args.rgb_image_path)
    args.segmentation_model_path = os.path.abspath(args.segmentation_model_path)
    if not os.path.exists(args.segmentation_model_path):
        print("Model path doesn't exist, please download the segmentation model. Exiting ...")
        exit()
    if args.compute_suction_pts:
        args.depth_image_path = os.path.abspath(args.depth_image_path)
        depth_image = cv2.imread(args.depth_image_path, -1)
        depth_image = depth_image/args.depth_scale
        c_matrix = [float(i) for i in args.c_matrix]
        c_matrix = np.array(c_matrix).reshape((3, 3))
    if args.use_buffered_gallery:
        args.gallery_buffered_path = os.path.abspath(args.gallery_buffered_path)
    elif not args.use_buffered_gallery:
        args.gallery_buffered_path = None

    print("Segmenting image")
    rgb_img = cv2.imread(args.rgb_image_path)
    segmentor= UnseenSegment(args.segmentation_model_path, device=device)
    seg_predictions = segmentor.segment_image(rgb_img)
    seg_img = draw_segmented_image(rgb_img, seg_predictions)

    cv2.imshow('Unseen object segmentation', seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.detect_all_objects:
        print("Classifying all objects")
        zero_shot_classifier = ZeroShotClassification(device=device, gallery_images_path=args.gallery_images_path, gallery_buffered_path=args.gallery_buffered_path, method=args.classification_method, siamese_model_path=os.path.abspath(args.siamese_model_path))
        class_predictions = zero_shot_classifier.classify_all_objects(rgb_img, seg_predictions)
        classified_image = draw_segmented_image(rgb_img, class_predictions, classes=os.listdir(args.gallery_images_path))

        cv2.imshow('Classify all objects from gallery', classified_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #zero_shot_classifier.save_gallery(PATH)

    if args.detect_one_object:
        obj_name = args.object_name
        print("Searching for object {}".format(obj_name))
        zero_shot_classifier = ZeroShotClassification(device=device, gallery_images_path=args.gallery_images_path, gallery_buffered_path=args.gallery_buffered_path, method=args.classification_method, siamese_model_path=os.path.abspath(args.siamese_model_path))
        class_predictions = zero_shot_classifier.find_object(rgb_img, seg_predictions, obj_name=obj_name)
        classified_image = draw_segmented_image(rgb_img, class_predictions, classes=[obj_name])

        cv2.imshow('Find a specific object', classified_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # show suction points for all objects
    if args.compute_suction_pts:
        print("Computing suction grasps for all objects")
        objects_point_clouds = compute_grasp.make_predicted_objects_clouds(rgb_img, depth_image, c_matrix, seg_predictions)
        suction_pts = compute_grasp.compute_suction_points(seg_predictions, objects_point_clouds)
        suction_pts_image = compute_grasp.visualize_suction_points(seg_img, c_matrix, suction_pts)

        cv2.imshow('Suction points for all objects', suction_pts_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
