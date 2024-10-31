#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import dounseen.core, dounseen.utils

import torch
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser()
    # input image and gallery
    parser.add_argument("--rgb-image-path", type=str, help="path rgb to image", default='./demo/scene_images/example.jpg')
    parser.add_argument('--gallery-images-path', type=str, help='path to gallery images folder', default='./demo/objects_gallery')
    # model paths
    parser.add_argument("--sam-model-path", type=str, help="path to unseen object segmentation model", default='./models/sam2/sam2_hiera_tiny.pt')  # TODO add instruction to download model
    parser.add_argument("--classification-model-path", type=str, help="path to unseen object classification model", default='./models/dounseen/vit_b_16_epoch_199_augment.pth')
    # Classifications parameters
    parser.add_argument("--classification-threshold", type=float, help="threshold for classification of all objects", default=0.6)
    parser.add_argument('--object-name', type=str, help='name of object (folder) to be detected when detecting one object', default='obj_000001')
    parser.add_argument('--classification-batch-size', type=int, help='batch size for classification', default=80)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No GPU found, this package is not optimized for CPU. Exiting ...")
        exit()
    else:
        device = torch.device("cuda")

    # get absolute paths
    args.rgb_image_path = os.path.abspath(args.rgb_image_path)
    args.gallery_images_path = os.path.abspath(args.gallery_images_path)
    args.sam_model_path = os.path.abspath(args.sam_model_path)
    args.classification_model_path = os.path.abspath(args.classification_model_path)

    # load rgb image
    rgb_img = Image.open(args.rgb_image_path)
    rgb_img = np.array(rgb_img.convert("RGB"))

    print("Create and run Sam2")
    sam2_mask_generator = create_sam2(args.sam_model_path, device=device)
    sam2_output = sam2_mask_generator.generate(rgb_img)
    sam2_masks, sam2_bboxes = dounseen.utils.reformat_sam2_output(sam2_output)

    sam2_segmentation_image = dounseen.utils.draw_segmented_image(rgb_img, sam2_masks, sam2_bboxes)
    dounseen.utils.show_wait_destroy("Unseen object segmentation", cv2.resize(sam2_segmentation_image, (0, 0), fx=0.5, fy=0.5))

    # create DoUnseen classifier
    segments = dounseen.utils.get_image_segments_from_binary_masks(rgb_img, sam2_masks, sam2_bboxes)  # get image segments from rgb image
    unseen_classifier = dounseen.core.UnseenClassifier(
        model_path=args.classification_model_path,
        gallery_images=args.gallery_images_path,
        gallery_buffered_path=None,
        augment_gallery=False,
        batch_size=args.classification_batch_size,
    )

    # Extracted gallery features can be saved and loaded for future use
    #unseen_classifier.save_gallery(PATH)

    # find one object
    matched_query, score = unseen_classifier.find_object(segments, obj_name=args.object_name, method="max")
    matched_query_ann_image = dounseen.utils.draw_segmented_image(rgb_img,
                                                                  [sam2_masks[matched_query]],
                                                                  [sam2_bboxes[matched_query]], classes_predictions=[0],
                                                                  classes_names=["obj_000001"])
    matched_query_ann_image = cv2.cvtColor(matched_query_ann_image, cv2.COLOR_RGB2BGR)
    dounseen.utils.show_wait_destroy("Find a specific object", cv2.resize(matched_query_ann_image, (0, 0), fx=0.5, fy=0.5))

    print("Classifying all objects")
    class_predictions, class_scores = unseen_classifier.classify_all_objects(segments, threshold=args.classification_threshold)
    filtered_class_predictions, filtered_masks, filtered_bboxes = dounseen.utils.remove_unmatched_query_segments(class_predictions, sam2_masks, sam2_bboxes)

    classified_image = dounseen.utils.draw_segmented_image(rgb_img, filtered_masks, filtered_bboxes, filtered_class_predictions, classes_names=os.listdir(args.gallery_images_path))
    classified_image = cv2.cvtColor(classified_image, cv2.COLOR_RGB2BGR)
    dounseen.utils.show_wait_destroy("Classify all objects from gallery", cv2.resize(classified_image, (0, 0), fx=0.5, fy=0.5))



def create_sam2(sam2_model_path, device):
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2 = build_sam2("sam2_hiera_t.yaml", sam2_model_path, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model = sam2,
        points_per_side=20,
        points_per_batch=20,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        multimask_output=False,
    )
    return mask_generator


if __name__ == '__main__':
    main()
