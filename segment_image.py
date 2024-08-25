#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse

from dounseen.dounseen import UnseenClassifier
from dounseen import utils

import torch
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-image-path", type=str, help="path rgb to image", default='../demo/rgb_images/000000.png')
    parser.add_argument("--maskrcnn-model-path", type=str, help="path to unseen object segmentation model", default='../models/segmentation/segmentation_mask_rcnn.pth')
    parser.add_argument("--sam-model-path", type=str, help="path to unseen object segmentation model", default='../models/sam_vit_b_01ec64.pth')  # TODO add instruction to download model
    parser.add_argument("--filter-sam-predictions", dest='filter_sam_predictions', action='store_true')
    parser.add_argument("--smallest-segment-size", type=int, help="smallest segment size to keep", default=50*50)
    parser.set_defaults(filter_sam_predictions=True)

    parser.add_argument('--classification-method', type=str, help="method to use for classification 'resnet-50-ctl' or 'vit-b-16-ctl'", default='vit-b-16-ctl')
    parser.add_argument("--classification-threshold", type=float, help="threshold for classification", default=0.5)
    parser.add_argument("--classification-model-path", type=str, help="path to unseen object classification model", default='../models/classification/classification_vit_b_16_ctl.pth')
    parser.add_argument('--detect-all-objects', dest='detect_all_objects', action='store_true')
    parser.set_defaults(detect_all_objects=True)
    parser.add_argument('--detect-one-object', dest='detect_one_object', action='store_true')
    parser.set_defaults(detect_one_object=True)
    parser.add_argument('--object-name', type=str, help='name of object (folder) to be detected', default='obj_000025')
    parser.add_argument('--gallery-images-path', type=str, help='path to gallery images folder', default='../demo/object_gallery_real_resized_256')
    parser.add_argument('--use-buffered-gallery', dest='use-buffered-gallery', action='store_true')
    parser.set_defaults(use_buffered_gallery=False)
    parser.add_argument('--gallery_buffered_path', type=str, help='path to buffered gallery file', default='../demo/??????.pkl')

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
    segmentor = UnseenSegment(
        method=args.segmentation_method,
        sam_model_path=args.sam_model_path,
        maskrcnn_model_path=args.maskrcnn_model_path,
        filter_sam_predictions=args.filter_sam_predictions,
        smallest_segment_size=args.smallest_segment_size
    )
    seg_predictions = segmentor.segment_image(rgb_img)
    seg_img = utils.draw_segmented_image(rgb_img, seg_predictions)

    utils.show_wait_destroy("Unseen object segmentation", cv2.resize(seg_img, (0, 0), fx=0.5, fy=0.5))

    if args.detect_all_objects or args.detect_one_object:
        # get image segments from rgb image
        segments = UnseenSegment.get_image_segments_from_binary_masks(rgb_img, seg_predictions)
        unseen_classifier = UnseenClassifier(
            model_path=args.classification_model_path,
            gallery_images=args.gallery_images_path,
            gallery_buffered_path=args.gallery_buffered_path,
            augment_gallery=True,
            method=args.classification_method,
            batch_size=args.batch_size,
        )
        #unseen_classifier.save_gallery(PATH)

    if args.detect_all_objects:
        print("Classifying all objects")
        class_predictions, class_scores = unseen_classifier.classify_all_objects(segments, threshold=args.classification_threshold)

        # remove class predictions with class -1 and their corresponding segments
        new_seg_predictions = []
        new_class_predictions = []
        for idx in range(len(class_predictions)):
            if class_predictions[idx] != -1:
                new_seg_predictions.append(idx)
                new_class_predictions.append(class_predictions[idx])
        new_seg_predictions = {'masks': [seg_predictions['masks'][i] for i in new_seg_predictions],
                               'bboxes': [seg_predictions['bboxes'][i] for i in new_seg_predictions]}

        classified_image = utils.draw_segmented_image(rgb_img, new_seg_predictions, new_class_predictions, classes_names=os.listdir(args.gallery_images_path))

        utils.show_wait_destroy("Classify all objects from gallery", cv2.resize(classified_image, (0, 0), fx=0.5, fy=0.5))

    if args.detect_one_object:
        obj_name = args.object_name
        print("Searching for object {}".format(obj_name))
        class_prediction = unseen_classifier.find_object(segments, obj_name=obj_name, method='max')
        class_seg_prediction = {'masks': [seg_predictions['masks'][class_prediction[0]]], 'bboxes': [seg_predictions['bboxes'][class_prediction[0]]]}
        class_prediction = [0]  # only one mask in the visualization
        classified_image = utils.draw_segmented_image(rgb_img, class_seg_prediction, classes_predictions=class_prediction, classes_names=[obj_name])

        utils.show_wait_destroy("Find a specific object", cv2.resize(classified_image, (0, 0), fx=0.5, fy=0.5))


if __name__ == '__main__':
    main()
