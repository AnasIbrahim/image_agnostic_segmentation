#!/usr/bin/env python
import os
import numpy as np
import torch
from PIL import Image
from dounseen import dounseen, utils
import cv2

device = torch.device("cuda")

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

image = Image.open('/home/gouda/Downloads/000000.png')
#image = Image.open('/home/gouda/Downloads/dortbunt_extra_sachen.jpg')
image = np.array(image.convert("RGB"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2 = build_sam2("sam2_hiera_t.yaml", "/home/gouda/segmentation/dounseen/models/sam2/sam2_hiera_tiny.pt", device=device)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    point_per_side=32,
    points_per_batch=32,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    mask_threshold=0.0,
    box_nms_thresh=0.7,
    crop_n_layers=0,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=512 / 1500,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=50,  # changed
    use_m2m=False,
    multimask_output=True,
)

anns = mask_generator.generate(image)

masks = [ann['segmentation'] for ann in anns]

bboxes = [ann['bbox'] for ann in anns]
bboxes = [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] for bbox in bboxes]
# change bboxed from xywh to xyxy
bboxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bboxes]

# dummy class predictions and names - same length as masks
classes_predictions = range(len(masks))
classes_names = [str(i) for i in classes_predictions]

# make a dict with masks and bboxes
anns_image = utils.draw_segmented_image(image, masks, bboxes, classes_predictions=classes_predictions, classes_names=classes_names)
#anns_image = cv2.cvtColor(anns_image, cv2.COLOR_RGB2BGR)
#utils.show_wait_destroy("Unseen object segmentation", cv2.resize(anns_image, (0, 0), fx=0.5, fy=0.5))

gallery_path = "/home/gouda/segmentation/image_agnostic_segmentation/demo/objects_gallery"

unseen_classifier = dounseen.UnseenClassifier(
    model_path="/home/gouda/segmentation/dounseen/models/dounseen/vit_b_16_epoch_199_augment.pth",
    gallery_images=gallery_path,
    gallery_buffered_path=None,
    augment_gallery=False,
    batch_size=32,
)

segments = utils.get_image_segments_from_binary_masks(image, masks, bboxes)

matched_query, score = unseen_classifier.find_object(segments, obj_name="obj_000001", method="max")

matched_query_ann_image = utils.draw_segmented_image(image, [masks[matched_query]], [bboxes[matched_query]], classes_predictions=[0], classes_names=["obj_000001"])
matched_query_ann_image = cv2.cvtColor(matched_query_ann_image, cv2.COLOR_RGB2BGR)
utils.show_wait_destroy("Find a specific object", cv2.resize(matched_query_ann_image, (0, 0), fx=0.5, fy=0.5))
