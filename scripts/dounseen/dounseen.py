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

from . import utils


class UnseenSegment:
    def __init__(self, method='SAM', sam_model_path=None, maskrcnn_model_path=None, mask_rcnn_confidence=0.7, filter_sam_predictions=False, smallest_segment_size=None):
        self.method = method
        self.smallest_segment_size = smallest_segment_size
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            # throw exception no GPU found
            raise Exception("No GPU found, this package is not optimized for CPU.")

        if self.method == 'maskrcnn':
            self.predictor = self.make_maskrcnn_predictor(maskrcnn_model_path, mask_rcnn_confidence, device)
        elif self.method == 'SAM':
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            sam = sam_model_registry["vit_b"](checkpoint=sam_model_path)
            sam.to(device=device)
            self.filter_sam_predictions = filter_sam_predictions
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=70,
                points_per_batch=64,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95-0.1,
                stability_score_offset=1.0,
                box_nms_thresh=0.7,
                crop_n_layers=0,
                crop_nms_thresh=0.7,
                crop_overlap_ratio=512 / 1500,
                crop_n_points_downscale_factor=1,
                point_grids=None,
                min_mask_region_area=500,
                output_mode="coco_rle"
            )
            if self.filter_sam_predictions:
                # make maskrcnn predictor to filter background
                self.make_maskrcnn_predictor(maskrcnn_model_path, mask_rcnn_confidence, device)
        else:
            # throw exception method doesn't exist
            raise Exception("Invalid segmentation method")

    def segment_image(self, img):
        if self.method == 'maskrcnn':
            detectron2_predictions = self.predictor(img)
            maskrcnn_predictions = self.formulate_maskrcnn_predictions(detectron2_predictions)
            return maskrcnn_predictions
        elif self.method == 'SAM':
            masks = self.mask_generator.generate(img)
            masks = sorted(masks, key=(lambda x: x['area']), reverse=True)  # smaller masks to be merged first
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
                # convert bbox to int
                bbox = [int(val) for val in bbox]
                bboxes.append(bbox)
            sam_predictions = {'masks': binary_masks, 'bboxes': bboxes}
            if self.filter_sam_predictions:
                # get maskrcnn predictions
                detectron2_predictions = self.predictor(img)
                maskrcnn_predictions = self.formulate_maskrcnn_predictions(detectron2_predictions)
                sam_predictions = self.sam_filter_background(maskrcnn_predictions, sam_predictions)
                #sam_predictions = self.sam_merge_small_masks(sam_predictions)  # TODO this would keep undersegmented masks so it is better to not use it and depend on the classification threshold to remove small masks
            if self.smallest_segment_size is not None:
                # filter out small masks
                filtered_masks = []
                filtered_bboxes = []
                for idx, mask in enumerate(sam_predictions['masks']):
                    if np.sum(mask) > self.smallest_segment_size:
                        filtered_masks.append(mask)
                        filtered_bboxes.append(sam_predictions['bboxes'][idx])
                sam_predictions = {'masks': filtered_masks, 'bboxes': filtered_bboxes}
            return sam_predictions

    def make_maskrcnn_predictor(self, maskrcnn_model_path, mask_rcnn_confidence=0.7, device='cuda'):
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg

        # --- detectron2 Config setup ---
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # WAS 3x.y
        cfg.MODEL.WEIGHTS = maskrcnn_model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
        cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = mask_rcnn_confidence
        cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(cfg)
        return self.predictor

    def formulate_maskrcnn_predictions(self, predictions):
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
        predictions = {'masks': masks, 'bboxes': bboxes}
        return predictions

    def sam_filter_background(self, maskrcnn_predictions, sam_predictions):
        # combine maskrcnn predictions in one mask
        maskrcnn_mask = np.zeros(maskrcnn_predictions['masks'][0].shape, dtype=bool)
        for mask in maskrcnn_predictions['masks']:
            maskrcnn_mask = np.logical_or(maskrcnn_mask, mask)
        # filter sam predictions using maskrcnn mask and keep only the ones that are (mostly) inside maskrcnn mask
        filtered_masks = []
        filtered_bboxes = []
        for idx, mask in enumerate(sam_predictions['masks']):
            # if 90% of sam mask is inside maskrcnn mask, then it is a valid prediction
            if np.sum(np.logical_and(mask, maskrcnn_mask)) / np.sum(mask) > 0.9:
                filtered_masks.append(mask)
                filtered_bboxes.append(sam_predictions['bboxes'][idx])
        sam_predictions = {'masks': filtered_masks, 'bboxes': filtered_bboxes}
        return sam_predictions

    def sam_merge_small_masks(self, sam_predictions):
        '''
        if a mask is mostly (>90%) inside another mask, then keep the bigger mask
        This is due to segment anything generating small masks inside detected bigger objects
        function assumes that masks are sorted by area
        '''
        # TODO current logic would not handle 3 masks hierarchy
        masks = sam_predictions['masks']
        bboxes = sam_predictions['bboxes']
        kept_masks = []
        removed_masks = []
        # iterate over masks from biggest to smallest
        for idx in range(len(masks)-1):
            mask = masks[idx]
            # iterate over the rest of the masks
            for idx2 in range(idx+1, len(masks)):
                mask2 = masks[idx2]
                # if mask2 is mostly (>90%) inside mask, then remove it
                if np.sum(np.logical_and(mask2, mask)) / np.sum(mask2) > 0.9:
                    if idx not in removed_masks:
                        kept_masks.append(idx)
                        removed_masks.append(idx2)
            # if there are no masks inside mask, then add mask to kept masks
            if idx not in removed_masks:
                kept_masks.append(idx)
        # if last mask is not removed, then add it to kept masks
        if len(masks) not in removed_masks:
            kept_masks.append(len(masks)-1)
        kept_masks = np.unique(kept_masks)
        # replace mask with merged mask
        masks = [masks[idx] for idx in kept_masks]
        bboxes = [bboxes[idx] for idx in kept_masks]
        sam_predictions = {'masks': masks, 'bboxes': bboxes}
        return sam_predictions

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
    def find_object(self, segments, obj_name, method='max'):
        query_feats = self.extract_query_feats(segments)
        obj_feats = self.gallery_feats[np.where(self.gallery_classes == self.gallery_obj_names.index(obj_name))[0]]
        if method == 'centroid':
            # calculate centroid of gallery object features
            obj_feats = torch.mean(obj_feats, dim=0)
            # calculate cosine similarity between query and gallery objects using torch cdist
            dist_matrix = torch.nn.functional.cosine_similarity(query_feats, obj_feats.unsqueeze(0), dim=1)
            dist_matrix = 1 - dist_matrix
            # find the id of the closest distance from the 1-D array
            matched_query = torch.where(dist_matrix == torch.min(dist_matrix))[0]
            print(torch.argsort(dist_matrix.squeeze()))
        elif method == 'weighted-max':
            # calculate cosine similarity between query and gallery objects using torch cdist
            dist_matrix = torch.nn.functional.cosine_similarity(query_feats.unsqueeze(0), obj_feats.unsqueeze(1), dim=2)
            dist_matrix = 1 - dist_matrix
            dist_matrix = dist_matrix.transpose(0, 1)
            # get sorted idx of closest dists
            arr = dist_matrix.cpu().numpy()
            min_dists = np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape)).squeeze(0)[:, 0]
            # only use the top 20 closest dists
            top_count = 20
            min_dists = min_dists[:top_count]
            # find unique values in min_dists
            unique, counts = np.unique(min_dists, return_counts=True)
            # calculate weighted score for each unique value: score = sum(index) * value / count
            scores = []
            for idx, unique_val in enumerate(unique):
                score = np.sum(np.where(min_dists == unique_val)) * unique_val / counts[idx]
                scores.append(score)
            # highest score is the matched query
            matched_query = unique[np.argmin(scores)]
            # if matched query is not unique, choose the first one
            if len(np.where(scores == np.min(scores))[0]) > 1:
                matched_query = np.where(scores == np.min(scores))[0][0]
        elif method == 'max':
            # calculate cosine similarity between query and gallery objects using torch cdist
            dist_matrix = torch.nn.functional.cosine_similarity(query_feats.unsqueeze(0), obj_feats.unsqueeze(1), dim=2)
            dist_matrix = 1 - dist_matrix
            dist_matrix = dist_matrix.transpose(0, 1)
            # find the id of the closest distance from the 2D array
            matched_query = torch.where(dist_matrix == torch.min(dist_matrix))[0]
            # if matched query is not unique, choose the first one
            if len(matched_query) > 1:
                matched_query = matched_query[0]
            score = torch.min(dist_matrix)
        else:
            raise Exception("Invalid classificaiton method. Use 'max', 'centroid' or 'trimmed-mean'")

        # if dimension of matched query is not 1, then choose the first one
        # this will happen when 2 vectors have the exact same distance, but this would rarely rarely happen
        #if matched_query.shape[0] > 1:
        #    #print("Warning: more than one image had the exact same score. Matched query: " + str(matched_query))
        #    matched_query = matched_query[0]
        # TODO when this happens take choose object with closer centroid

        return matched_query, score

    def classify_all_objects(self, segments, threshold=0.6):
        query_feats = self.extract_query_feats(segments)
        obj_feats = self.gallery_feats
        # calculate cosine similarity between query and gallery objects using torch cosine_similarity
        dist_matrix = torch.nn.functional.cosine_similarity(query_feats.unsqueeze(0), obj_feats.unsqueeze(1), dim=2)
        dist_matrix = dist_matrix.transpose(0, 1)
        # find the closest gallery object for each query object
        matches = torch.max(dist_matrix, dim=1)
        matched_queries = matches.indices.detach().cpu().numpy()
        pred_scores = matches.values.detach().cpu().numpy()
        # convert matched queries to gallery classes
        pred_classes = self.gallery_classes[matched_queries]
        # set predictions with distance less than threshold to -1
        pred_classes[np.where(pred_scores < threshold)] = -1
        return pred_classes, pred_scores

    @torch.no_grad()
    def extract_query_feats(self, segments):
        query_imgs = copy.deepcopy(segments)
        query_imgs = [utils.resize_and_pad(query_img, (224,224), (255,255,255)) for query_img in query_imgs]
        query_imgs = torch.stack([self.transform(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)) for query_img in query_imgs])  # TODO that converting from BGR to RGB is correct
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
