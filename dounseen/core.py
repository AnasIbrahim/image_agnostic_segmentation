import copy
import glob
import os
import numpy as np
import torch
from PIL import Image
import pickle
import pkg_resources

import torchvision
from torchvision import transforms

IMAGE_SIZE = 384

class BackgroundFilter:
    def __init__(self, maskrcnn_model_path='DEFAULT', maskrcnn_confidence=0.7, background_filter_threshold=0.9):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            # throw exception no GPU found
            raise Exception("No GPU found, this package is not optimized for CPU.")

        if maskrcnn_model_path == 'DEFAULT':
            model_path = pkg_resources.resource_filename('dounseen', '../models/background_filtering/background_filter_Mask-RCNN.pth')
            # check if model file exists
            if not os.path.exists(model_path):
                raise Exception("Background filter model file not found in the default path. Was the model downloaded from HuggingFace?")
        else:
            model_path = maskrcnn_model_path

        self.predictor = self.make_maskrcnn_predictor(model_path, maskrcnn_confidence)
        self.background_filter_threshold = background_filter_threshold

    def filter_background_annotations(self, img, sam_masks, sam_bboxes):
        # get maskrcnn predictions
        detectron2_predictions = self.predictor(img)
        maskrcnn_masks, maskrcnn_bboxes = self.formulate_maskrcnn_predictions(detectron2_predictions)
        sam_predictions = self.remove_background_masks(maskrcnn_masks, maskrcnn_bboxes, sam_masks, sam_bboxes)
        return sam_predictions

    def make_maskrcnn_predictor(self, maskrcnn_model_path, maskrcnn_confidence=0.7):
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
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = maskrcnn_confidence
        cfg.MODEL.DEVICE = self.device
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
        return masks, bboxes

    def remove_background_masks(self, maskrcnn_masks, maskrcnn_bboxes, sam_masks, sam_bboxes):
        # combine maskrcnn predictions in one mask
        maskrcnn_mask = np.zeros(maskrcnn_masks[0].shape, dtype=bool)
        for mask in maskrcnn_masks:
            maskrcnn_mask = np.logical_or(maskrcnn_mask, mask)
        # filter sam predictions using maskrcnn mask and keep only the ones that are (mostly) inside maskrcnn mask
        filtered_masks = []
        filtered_bboxes = []
        for idx, mask in enumerate(sam_masks):
            # if 90% of sam mask is inside maskrcnn mask, then it is a valid prediction
            if np.sum(np.logical_and(mask, maskrcnn_mask)) / np.sum(mask) > self.background_filter_threshold:
                filtered_masks.append(mask)
                filtered_bboxes.append(sam_bboxes[idx])
        return filtered_masks, filtered_bboxes


class UnseenClassifier:
    '''
    classify unseen objects without requiring any training data.
    The problem is formulated as an image association between query and gallery images.
    The gallery images are manually pre-captured images of the objects to be classified.
    The query images depend on the application:
        1- Standalone: query images are images of the isolated object to be classified.
        2- Full segmentation: query images are segments from zero-shot segmentation models like Segment-Anything.
    Class requires CUDA enabled GPU.
    '''
    def __init__(self, classification_model_path='DEFAULT', gallery_images=None, gallery_buffered_path=None, augment_gallery=False, batch_size=32):
        '''
        Arguments:
            classification_model_path: str
                  path to the classification model. If 'DEFAULT', the default model will be loaded.
            gallery_images: str or dict
                path to the gallery images or dictionary of gallery images (dict format: {obj_name: [list of PIL images]})
            gallery_buffered_path: str
                path to the buffered gallery features. If provided, gallery images will not be used.
            augment_gallery: bool
                if True, augment gallery images by rotating them 8 times (0, 45, 90, 135, 180, 225, 270, 315 degrees)
            batch_size: int
                batch size depends on the GPU memory, default is 32
        '''
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            # throw exception no GPU found
            raise Exception("No GPU found, this package is not optimized for CPU.")

        self.augment_gallery = augment_gallery
        self.batch_size = batch_size
        # load model weights
        if classification_model_path == 'DEFAULT':
            model_path = pkg_resources.resource_filename('dounseen', '../models/dounseen/vit_b_16_epoch_199_augment.pth')
            # check if model file exists
            if not os.path.exists(model_path):
                raise Exception("Classification model file not found in the default path. Was the model downloaded from HuggingFace?")
        else:
            model_path = classification_model_path
        model_weights = torch.load(model_path, map_location=self.device)
        # TODO load SWAG model without downloading
        # load IMAGENET1K_SWAG_E2E_V1
        self.model_backbone = torchvision.models.vit_b_16(weights='IMAGENET1K_SWAG_E2E_V1')
        self.model_backbone.load_state_dict(model_weights)
        self.model_backbone.to(self.device)

        self.feed_shape = [3, IMAGE_SIZE, IMAGE_SIZE]
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
        '''
        Save gallery features to a file. The file can be loaded later to avoid recomputing gallery features.
        '''
        gallery_dict = {'gallery_obj_names': self.gallery_obj_names,
                        'gallery_feats': self.gallery_feats,
                        'gallery_classes': self.gallery_classes}
        with open(path, 'wb') as f:
            pickle.dump(gallery_dict, f)

    @torch.no_grad()
    def update_gallery(self, gallery_images):
        '''
        Extract gallery features from gallery images
        Arguments:
            gallery_images: str or dict
                path to the gallery images or dictionary of gallery images (dict format: {obj_name: [list of PIL images]})
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

    def find_object(self, segments, obj_name, method='max'):
        '''
        Find an object in the gallery that matches the query object.
        The method find only a single instance of the object.
        The method will always return an object. So the object of interest is assumed to be in the gallery.

        Arguments:
            segments: list of PIL images
                list of query images
            obj_name: str
                name of the object to be found as in the gallery dictionary
            method: str
                method to use for classification. 'max' or 'centroid'
        '''
        query_feats = self.extract_query_feats(segments)
        obj_feats = self.gallery_feats[np.where(self.gallery_classes == self.gallery_obj_names.index(obj_name))[0]]
        if method == 'centroid':
            # calculate centroid of gallery object features
            obj_feats = torch.mean(obj_feats, dim=0)
            # calculate cosine similarity between query and gallery objects using torch cdist
            dist_matrix = torch.nn.functional.cosine_similarity(query_feats, obj_feats.unsqueeze(0), dim=1)
            # find the id of the highest similarity from the 1-D array
            matched_query = torch.where(dist_matrix == torch.max(dist_matrix))[0]
            score = torch.max(dist_matrix)
        elif method == 'max':
            # calculate cosine similarity between query and gallery objects using torch cdist
            dist_matrix = torch.nn.functional.cosine_similarity(query_feats.unsqueeze(0), obj_feats.unsqueeze(1), dim=2)
            dist_matrix = dist_matrix.transpose(0, 1)
            # find the id of the biggest similarity from the 2D array
            matched_query = torch.where(dist_matrix == torch.max(dist_matrix))[0]
            score = torch.max(dist_matrix)
        else:
            raise Exception("Invalid classificaiton method. Use 'max' or 'centroid'")

        # if dimension of matched query is not 1, then 2 vectors have the exact same distance
        # this would rarely rarely happen
        # temporary solution just return the first object
        if len(matched_query) > 1:
            matched_query = matched_query[0]
        # TODO resolve using both centroid and max
        #if matched_query.shape[0] > 1:
        #    calculate using the other method

        return matched_query, score

    def classify_all_objects(self, segments, threshold=0.6, multi_instance=False):
        '''
        Classify all objects in the query images.
        Only associations with similarity above the threshold are considered.

        Arguments:
            segments: list of PIL images
                list of query images
            threshold: float
                threshold for similarity between query and gallery objects
            multi_instance: bool
                if True, keep multiple instances of the same class in the query images.
        '''
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
        if not multi_instance:  # keep the highest score for each class
            # find all unique classes
            unique_classes = np.unique(pred_classes)
            # remove -1 from unique classes
            unique_classes = unique_classes[unique_classes != -1]
            # for each unique class, keep the highest score and set the rest to -1
            for unique_class in unique_classes:
                class_indices = np.where(pred_classes == unique_class)
                class_scores = pred_scores[class_indices]
                max_score_index = np.argmax(class_scores)
                pred_classes[class_indices] = -1
                pred_classes[class_indices[0][max_score_index]] = unique_class
        return pred_classes, pred_scores

    @torch.no_grad()
    def extract_query_feats(self, segments):
        query_imgs = copy.deepcopy(segments)
        query_imgs = torch.stack([self.transform(query_img) for query_img in query_imgs])
        query_imgs = query_imgs.to(device=self.device)
        # split image to fit into GPU memory
        # probably number of query images is small enough to fit into GPU memory at once but just to be sure
        query_images = torch.split(query_imgs, self.batch_size)
        query_feats = []
        for batch in query_images:
            batch_feats = self.model_backbone(batch)
            query_feats.append(batch_feats)
        query_feats = torch.cat(query_feats)
        return query_feats
