#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse

from DoUnseen.Dounseen import UnseenSegment, UnseenClassifier, draw_segmented_image
from DoUnseen import compute_grasp

import torch
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-image-path", type=str, help="path rgb to image", default='../demo/rgb_images/000000.png')
    parser.add_argument("--segmentation-method", type=str, help="method to use for segmentation 'maskrcnn' or 'SAM' ", default='maskrcnn')
    parser.add_argument("--maskrcnn-model-path", type=str, help="path to unseen object segmentation model", default='../models/segmentation/segmentation_mask_rcnn.pth')
    parser.add_argument("--sam-model-path", type=str, help="path to unseen object segmentation model", default='/path/to/sam/model.pth')
    parser.add_argument("--filter-sam-predictions", dest='filter_sam_predictions', action='store_true')
    parser.set_defaults(filter_sam_predictions=True)

    parser.add_argument('--classification-method', type=str, help="method to use for classification 'resnet-50-ctl' or 'vit-b-16-ctl'", default='vit-b-16-ctl')
    parser.add_argument("--classification-model-path", type=str, help="path to unseen object classification model", default='../models/classification/classification_vit_b_16_ctl.pth')
    parser.add_argument('--detect-all-objects', dest='detect_all_objects', action='store_true')
    parser.set_defaults(detect_all_objects=True)
    parser.add_argument('--detect-one-object', dest='detect_one_object', action='store_true')
    parser.set_defaults(detect_one_object=True)
    parser.add_argument('--object-name', type=str, help='name of object (folder) to be detected', default='obj_000025')
    parser.add_argument('--gallery-images-path', type=str, help='path to gallery images folder', default='../demo/objects_gallery')
    parser.add_argument('--use-buffered-gallery', dest='use-buffered-gallery', action='store_true')
    parser.set_defaults(use_buffered_gallery=False)
    parser.add_argument('--gallery_buffered_path', type=str, help='path to buffered gallery file', default='../demo/??????.pkl')

    parser.add_argument('--compute-suction-pts', dest='compute_suction_pts', action='store_true')
    parser.set_defaults(compute_suction_pts=False)
    parser.add_argument("--depth-image-path", type=str, help="path to depth image", default='../demo/depth_images/000000.png')
    parser.add_argument("--depth-scale", type=int, help="depth image in divided by the scale", default=1000)  # convert from mm to meter - this is done due to depth images being saves in BOP format
    parser.add_argument('-c-matrix', nargs='+',
                        help='camera matrix to convert depth image to point cloud',
                        default=['1390.53', '0.0', '964.957', '0.0', '1386.99', '522.586', '0.0', '0.0', '1.0']) # HOPE dataset - example images

    parser.add_argument('--batch-size', type=int, help='batch size for classification', default=100)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No GPU found, this package is not optimized for CPU. Exiting ...")
        exit()

    if args.segmentation_method == 'maskrcnn':
        if os.path.exists(args.maskrcnn_model_path):
            args.maskrcnn_model_path = os.path.abspath(args.maskrcnn_model_path)
        else:
            print("Mask R-CNN model path doesn't exist, please download the segmentation model. Exiting ...")
            exit()
    elif args.segmentation_method == 'SAM':
        if os.path.exists(args.sam_model_path):
            args.sam_model_path = os.path.abspath(args.sam_model_path)
        else:
            print("SAM model path doesn't exist, please download the segmentation model. Exiting ...")
            exit()

    if args.classification_method == 'resnet-50-ctl':
        if os.path.exists(args.classification_model_path):
            args.classification_model_path = os.path.abspath(args.classification_model_path)
        else:
            print("ResNet-50 model path doesn't exist, please download the classification model. Exiting ...")
            exit()
    elif args.classification_method == 'vit-b-16-ctl':
        if os.path.exists(args.classification_model_path):
            args.classification_model_path = os.path.abspath(args.classification_model_path)
        else:
            print("ViT-B/16 model path doesn't exist, please download the classification model. Exiting ...")
            exit()

    # get absolute paths
    args.rgb_image_path = os.path.abspath(args.rgb_image_path)
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
    segmentor = UnseenSegment(method=args.segmentation_method, sam_model_path=args.sam_model_path, maskrcnn_model_path=args.maskrcnn_model_path)
    seg_predictions = segmentor.segment_image(rgb_img)
    seg_img = draw_segmented_image(rgb_img, seg_predictions)

    cv2.imshow('Unseen object segmentation', cv2.resize(seg_img, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.detect_all_objects or args.detect_one_object:
        # get image segments from rgb image
        segments = UnseenSegment.get_image_segments_from_binary_masks(rgb_img, seg_predictions)
        unseen_classifier = UnseenClassifier(model_path=args.classification_model_path, gallery_images=args.gallery_images_path, gallery_buffered_path=args.gallery_buffered_path, augment_gallery=True, method=args.classification_method, batch_size=args.batch_size)
        #unseen_classifier.save_gallery(PATH)

    if args.detect_all_objects:
        print("Classifying all objects")
        class_predictions = unseen_classifier.classify_all_objects(segments, centroid=False)
        classified_image = draw_segmented_image(rgb_img, seg_predictions, class_predictions, classes_names=os.listdir(args.gallery_images_path))

        cv2.imshow('Classify all objects from gallery', cv2.resize(classified_image, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.detect_one_object:
        obj_name = args.object_name
        print("Searching for object {}".format(obj_name))
        class_prediction = unseen_classifier.find_object(segments, obj_name=obj_name, centroid=False)
        class_seg_prediction = {'masks': [seg_predictions['masks'][class_prediction[0]]], 'bboxes': [seg_predictions['bboxes'][class_prediction[0]]]}
        class_prediction = [0]  # only one mask in the visualization
        classified_image = draw_segmented_image(rgb_img, class_seg_prediction, classes_predictions=class_prediction, classes_names=[obj_name])

        cv2.imshow('Find a specific object', cv2.resize(classified_image, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # show suction points for all objects
    if args.compute_suction_pts:
        print("Computing suction grasps for all objects")
        objects_point_clouds = compute_grasp.make_predicted_objects_clouds(rgb_img, depth_image, c_matrix, seg_predictions)
        suction_pts = compute_grasp.compute_suction_points(seg_predictions, objects_point_clouds)
        suction_pts_image = compute_grasp.visualize_suction_points(seg_img, c_matrix, suction_pts)

        cv2.imshow('Suction points for all objects', cv2.resize(suction_pts_image, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
