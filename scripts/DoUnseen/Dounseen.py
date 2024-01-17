import copy
import glob
import os
import numpy as np
import cv2
import torch
from PIL import Image
import pickle
import pycocotools

import torchvision
from torchvision import transforms
import torch.nn.functional as F


def draw_segmented_image(rgb_img, seg_predictions, classes_predictions=None, classes_names=None):
    """
    draws a list of binary masks over an image.
    If predicted classes are provided, the masks are colored according to the class.
    """
    img = copy.deepcopy(rgb_img)
    masks = seg_predictions['masks']
    bboxes = seg_predictions['bboxes']
    opacity = 0.5
    for idx in range(len(masks)):
        mask = masks[idx]
        bbox = bboxes[idx]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # add mask to image with opacity
        img[mask] = ((1 - opacity) * img[mask] + opacity * np.array(color)).astype(np.uint8)
        # add a black border around the mask (using find contours and draw contours)
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 2)
        # add bbox (unfilled) to image
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    # add text to image in a separate loop to make text more visible
    for idx in range(len(bboxes)):
        bbox = bboxes[idx]
        if classes_predictions is not None:
            class_id = classes_predictions[idx]
            if classes_names is not None:
                class_name = classes_names[class_id]
            else:
                class_name = str(class_id)
            # add a blacked filled rectangle of the top left corner of the bbox that would fit the class name text
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + len(class_name) * 12, bbox[1] + 20), (0, 0, 0), -1)
            # add class name inside the rectangle without opacity
            cv2.putText(img, class_name, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


class UnseenSegment:
    def __init__(self, method='SAM', sam_model_path=None, maskrcnn_model_path=None, mask_rcnn_confidence=0.7, filter_sam_predictions=False):
        self.method = method
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            # throw exception no GPU found
            raise Exception("No GPU found, this package is not optimized for CPU.")

        if self.method == 'maskrcnn':
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg

            # --- detectron2 Config setup ---
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # WAS 3x.y
            cfg.MODEL.WEIGHTS = maskrcnn_model_path
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
            cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
            cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = mask_rcnn_confidence
            cfg.MODEL.DEVICE = device
            self.predictor = DefaultPredictor(cfg)
        elif self.method == 'SAM':
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            sam = sam_model_registry["vit_b"](checkpoint=sam_model_path)
            sam.to(device=device)
            self.filter_sam_predictions = filter_sam_predictions
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=50,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Requires open-cv to run post-processing,
                output_mode='coco_rle'
            )
        else:
            # throw exception method doesn't exist
            raise Exception("Invalid segmentation method")

    def segment_image(self, img):
        if self.method == 'maskrcnn':
            predictions = self.predictor(img)
            # make a list of binary masks from the predictions
            masks = predictions['instances'].pred_masks.cpu().numpy()
            masks = [mask.astype(bool) for mask in masks]
            # make a list of bounding boxes from the predictions
            bboxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
            bboxes = [[int(val) for val in bbox] for bbox in bboxes]
            # sort masks and bboxes by mask area
            areas = [np.sum(mask) for mask in masks]
            masks = [mask for _, mask in sorted(zip(areas, masks), key=lambda pair: pair[0], reverse=True)]
            bboxes = [bbox for _, bbox in sorted(zip(areas, bboxes), key=lambda pair: pair[0], reverse=True)]
            return {'masks': masks, 'bboxes': bboxes}

        elif self.method == 'SAM':
            masks = self.mask_generator.generate(img)
            binary_masks = []
            bboxes = []
            # convert coco rle mask to binary mask
            for mask in masks:
                binary_mask = pycocotools.mask.decode(mask['segmentation'])
                # convert mask to bool
                binary_mask = binary_mask.astype(bool)
                binary_masks.append(binary_mask)
                # convet bbox from XYWH to X1Y1X2Y2
                bbox = [mask['bbox'][0], mask['bbox'][1], mask['bbox'][0]+mask['bbox'][2], mask['bbox'][1]+mask['bbox'][3]]
                bboxes.append(bbox)
            masks = sorted(masks, key=(lambda x: x['area']), reverse=False)  # smaller masks to be merged first
            if self.filter_sam_predictions:
                from .utils import merge_masks
                binary_masks = merge_masks(masks, img)

            return {'masks': binary_masks, 'bboxes': bboxes}

    @staticmethod
    def get_image_segments_from_binary_masks(rgb_image, seg_predictions):
        masks = seg_predictions['masks']
        bboxes = seg_predictions['bboxes']
        segments = []
        for idx in range(len(masks)):
            mask = masks[idx]
            bbox = bboxes[idx]
            # add mask to the image with background being white
            segment = np.ones(rgb_image.shape, dtype=np.uint8) * 255
            segment[mask] = rgb_image[mask]
            segment = segment[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            segments.append(segment)
        return segments


class UnseenClassifier:
    def __init__(self, model_path, gallery_images=None, gallery_buffered_path=None, augment_gallery=False, method='vit-b-16-ctl', batch_size=32):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            # throw exception no GPU found
            raise Exception("No GPU found, this package is not optimized for CPU.")

        self.method = method
        self.augment_gallery = augment_gallery
        self.batch_size = batch_size
        # load model weights
        model_weights = torch.load(model_path, map_location=self.device)
        if method == 'vit-b-16-ctl':
            self.model_backbone = torchvision.models.vit_b_16(weights=None)
        elif method == 'resnet-50-ctl':
            self.model_backbone = torchvision.models.resnet50(weights=None)
        else:
            raise Exception("Invalid classification method")
        self.model_backbone.load_state_dict(model_weights)
        self.model_backbone.to(self.device)

        self.feed_shape = [3, 224, 224]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(self.feed_shape[1:], antialias=True)
        ])

        if gallery_buffered_path is not None:
            with open(gallery_buffered_path, 'rb') as f:
                gallery_dict = pickle.load(f)
            self.gallery_obj_names = gallery_dict['gallery_obj_names']
            self.gallery_feats = gallery_dict['gallery_feats']
            self.gallery_classes = gallery_dict['gallery_classes']

            self.gallery_feats = self.gallery_feats.to(device=self.device)
        elif gallery_images is not None:
            self.update_gallery(gallery_images)
        else:
            print("Warning: no gallery images or buffered gallery features are provided. \n"
                  "use update_gallery() to update gallery features")

    def save_gallery(self, path):
        gallery_dict = {'gallery_obj_names': self.gallery_obj_names,
                        'gallery_feats': self.gallery_feats,
                        'gallery_classes': self.gallery_classes}
        with open(path, 'wb') as f:
            pickle.dump(gallery_dict, f)

    @torch.no_grad()
    def update_gallery(self, gallery_images):
        '''
        Extract gallery features from gallery images
        :param gallery_images: path or dictionary of gallery images
        '''
        print("Extracting/updating gallery features")

        if isinstance(gallery_images, str):  # if path, load images and extract features
            gallery_path = gallery_images
            self.gallery_obj_names = os.listdir(gallery_path)
            gallery_dict = {}
            for obj_num, obj_name in enumerate(self.gallery_obj_names):
                obj_images_path = glob.glob(os.path.join(gallery_path, obj_name, '*'))
                obj_images = []
                for obj_image_path in obj_images_path:
                    obj_image = Image.open(obj_image_path).convert("RGB")
                    obj_images.append(obj_image)
                gallery_dict[obj_name] = obj_images
        elif isinstance(gallery_images, dict):  # if dictionary is list of lists of PIL images, extract features only
            gallery_dict = gallery_images
            self.gallery_obj_names = list(gallery_dict.keys())

        # Extract gallery images
        gallery_images = []
        self.gallery_feats = []
        self.gallery_classes = []
        for obj_num, obj_name in enumerate(gallery_dict):
            # TODO optimize code by feeding all images at once to self.transform
            obj_images = gallery_dict[obj_name]
            for obj_image in obj_images:
                if not self.augment_gallery:
                    gallery_images.append(self.transform(obj_image))
                    self.gallery_classes.append(obj_num)
                else:
                    # generate angles from 0 to 360 with step 45 degrees (remove last step)
                    angles = np.arange(0, 360, 45)
                    gallery_images.extend([self.transform(obj_image.rotate(angle, expand=True)) for angle in angles])
                    self.gallery_classes.extend([obj_num] * len(angles))  # multiply angles for augmentation
        self.gallery_classes = np.array(self.gallery_classes)

        gallery_images = torch.stack(gallery_images)
        gallery_images = gallery_images.to(device=self.device)
        # split image to fit into GPU memory
        gallery_images = torch.split(gallery_images, self.batch_size)
        self.gallery_feats = []
        for batch in gallery_images:
            batch_feats = self.model_backbone(batch)
            self.gallery_feats.append(batch_feats)
        self.gallery_feats = torch.cat(self.gallery_feats)
        #self.gallery_feats = F.normalize(self.gallery_feats, p=2, dim=1)

    @torch.no_grad()
    def find_object(self, segments, obj_name, centroid=False):
        query_feats = self.extract_query_feats(segments)
        obj_feats = self.gallery_feats[np.where(self.gallery_classes == self.gallery_obj_names.index(obj_name))[0]]
        if centroid:
            # calculate centroid of gallery object features
            obj_feats = torch.mean(obj_feats, dim=0)
            # calculate cosine similarity between query and gallery objects using torch cdist
            dist_matrix = torch.nn.functional.cosine_similarity(query_feats, obj_feats.unsqueeze(0), dim=1)
            dist_matrix = 1 - dist_matrix
            # find the id of the closest distance from the 1-D array
            matched_query = torch.where(dist_matrix == torch.min(dist_matrix))[0]
            print(torch.argsort(dist_matrix.squeeze()))
        else:
            # calculate cosine similarity between query and gallery objects using torch cdist
            dist_matrix = torch.nn.functional.cosine_similarity(query_feats.unsqueeze(0), obj_feats.unsqueeze(1), dim=2)
            dist_matrix = 1 - dist_matrix
            dist_matrix = dist_matrix.transpose(0, 1)
            # find the id of the closest distance from the 2D array
            matched_query = torch.where(dist_matrix == torch.min(dist_matrix))[0]

        return matched_query

    def classify_all_objects(self, segments, centroid=False):
        query_feats = self.extract_query_feats(segments)
        # if centroid is True, calculate centroid of gallery object features
        if not centroid:
            obj_feats = self.gallery_feats
        if centroid:
            obj_feats = []
            for obj_num, obj_name in enumerate(self.gallery_obj_names):
                obj_feats.append(torch.mean(self.gallery_feats[np.where(self.gallery_classes == obj_num)[0]], dim=0))
            obj_feats = torch.stack(obj_feats)
        # calculate cosine similarity between query and gallery objects using torch cosine_similarity
        dist_matrix = torch.nn.functional.cosine_similarity(query_feats.unsqueeze(0), obj_feats.unsqueeze(1), dim=2)
        dist_matrix = 1 - dist_matrix
        dist_matrix = dist_matrix.transpose(0, 1)
        # find the closest gallery object for each query object
        matched_queries = torch.argmin(dist_matrix, dim=1)
        matched_queries = matched_queries.cpu().numpy()
        # convert matched queries to gallery classes
        pred_classes = self.gallery_classes[matched_queries]
        return pred_classes

    @torch.no_grad()
    def extract_query_feats(self, segments):
        query_imgs = torch.stack([self.transform(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)) for segment in segments])  # TODO that converting from BGR to RGB is correct
        query_imgs = query_imgs.to(device=self.device)
        # split image to fit into GPU memory
        # probably number of query images is small enough to fit into GPU memory at once but just to be sure
        query_images = torch.split(query_imgs, self.batch_size)
        query_feats = []
        for batch in query_images:
            batch_feats = self.model_backbone(batch)
            query_feats.append(batch_feats)
        query_feats = torch.cat(query_feats)
        #query_feats = F.normalize(query_feats, p=2, dim=1)
        return query_feats
