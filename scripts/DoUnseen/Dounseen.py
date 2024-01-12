import copy
import glob
import os
import numpy as np
import cv2
import torch
from PIL import Image
from collections import OrderedDict
import pickle

import torchvision
from torchvision import transforms
import torch.nn.functional as F

def draw_segmented_image(img, predictions, classes=['']):
    # TODO draw masks witout dependency on detectron2
    try:
        from detectron2.data import MetadataCatalog
        from detectron2.utils.visualizer import Visualizer
        from detectron2.utils.visualizer import ColorMode
    except ImportError:
        print("Detectron2 is required for installation installed. Exiting ...")
        exit()

    if 'dounseen' in [x for x in MetadataCatalog]:
        MetadataCatalog.remove('dounseen')
    MetadataCatalog.get('dounseen').set(thing_classes=classes)
    metadata = MetadataCatalog.get('dounseen')
    v = Visualizer(img,
                   metadata=metadata,
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    img = out.get_image()

    return img


class UnseenSegment:
    def __init__(self, device, method='SAM', sam_model_path=None, maskrcnn_model_path=None, mask_rcnn_confidence=0.7, filter_sam_predictions=False):
        self.method = method

        if self.method == 'maskrcnn':
            from detectron2.utils.logger import setup_logger
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg

            setup_logger()  # initialize the detectron2 logger and set its verbosity level to “DEBUG”.
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
        elif self.method == 'SAM':
            # TODO remove dependency on detectron2
            from detectron2.structures.instances import Instances
            from detectron2.structures.boxes import Boxes
            import pycocotools
            masks = self.mask_generator.generate(img)
            # convert coco rle mask to binary mask
            for mask in masks:
                mask['segmentation'] = pycocotools.mask.decode(mask['segmentation'])
                # convert mask to bool
                mask['segmentation'] = mask['segmentation'].astype(bool)

                # convet bbox from XYWH to X1Y1X2Y2
                mask['bbox'] = [mask['bbox'][0], mask['bbox'][1], mask['bbox'][0]+mask['bbox'][2], mask['bbox'][1]+mask['bbox'][3]]

            masks = sorted(masks, key=(lambda x: x['area']), reverse=False)  # smaller masks to be merged first

            if self.filter_sam_predictions:
                from .utils import merge_masks
                masks = merge_masks(masks, img)

            predictions = Instances(
                image_size=(img.shape[0], img.shape[1]),
                pred_boxes=Boxes(torch.tensor([mask['bbox'] for mask in masks])),
                scores=torch.tensor([mask['stability_score'] for mask in masks]),
                pred_classes=torch.IntTensor(np.zeros(len(masks))),  # there are no classes - all clases are class 0
                pred_masks=torch.tensor(np.array([mask['segmentation'] for mask in masks]))
            )
            predictions = {'instances': predictions}
        return predictions


class UnseenClassifier:
    def __init__(self, device, model_path, gallery_images=None, gallery_buffered_path=None, augment_gallery=False, method='vit-b-16-ctl', batch_size=200):
        self.method = method
        self.device = device
        self.augment_gallery = augment_gallery
        self.batch_size = batch_size
        # load model weights
        model_weights = torch.load(model_path, map_location=device)
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
        gallery_dict = {}
        gallery_dict['gallery_obj_names'] = self.gallery_obj_names
        gallery_dict['gallery_feats'] = self.gallery_feats
        gallery_dict['gallery_classes'] = self.gallery_classes
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
                    angles = [0, 45, 90, 135, 180, 225, 270, 315]
                    gallery_images.extend([self.transform(obj_image.rotate(angle)) for angle in angles])
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
        self.gallery_feats = F.normalize(self.gallery_feats, p=2, dim=1)

    @torch.no_grad()
    def find_object(self, rgb_img, predictions, obj_name, centroid=False):
        query_feats = self.extract_query_feats(rgb_img, predictions)
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
        obj_predictions = copy.deepcopy(predictions)
        obj_predictions['instances'] = obj_predictions['instances'][matched_query]

        return obj_predictions

    def classify_all_objects(self, rgb_img, predictions):
        classified_predictions = copy.deepcopy(predictions)
        pred_classes = []
        query_feats = self.extract_query_feats(rgb_img, predictions)
        for query_id, query_feat in enumerate(query_feats):
            obj_dists = []
            for obj_name in self.gallery_obj_names:
                obj_feats = self.gallery_feats[np.where(self.gallery_classes == self.gallery_obj_names.index(obj_name))[0]]
                dists = torch.cosine_similarity(query_feat, obj_feats, dim=1)
                obj_dists.append(np.array(dists.cpu()))
            obj_dists = [max(obj) for obj in obj_dists]  # TODO check for a threshold where the object is not in query
            pred_class = obj_dists.index(max(obj_dists))
            pred_classes.append(pred_class)
        classified_predictions['instances'].pred_classes = torch.tensor(pred_classes)
        return classified_predictions

    @torch.no_grad()
    def extract_query_feats(self, rgb_img, predictions):
        query_images = []
        instances = predictions['instances'].to('cpu')
        for idx in range(len(instances)):
            bbox = instances[idx].pred_boxes.tensor.squeeze().numpy()
            mask = instances[idx].pred_masks.squeeze().numpy().astype(np.uint8)
            # make a masked rgb image with a white background
            masked_rgb = np.ones(rgb_img.shape, dtype=np.uint8) * 255
            masked_rgb[mask == 1] = rgb_img[mask == 1]
            bbox = [int(val) for val in bbox]
            obj_cropped_mask = masked_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            #cv2.imshow('image', obj_cropped_mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            query_img = self.transform(cv2.cvtColor(obj_cropped_mask, cv2.COLOR_BGR2RGB))
            query_images.append(query_img)
        query_images = torch.stack(query_images)
        query_images = query_images.to(device=self.device)
        # split image to fit into GPU memory
        # probably number of query images is small enough to fit into GPU memory at once but just to be sure
        query_images = torch.split(query_images, self.batch_size)
        query_feats = []
        for batch in query_images:
            batch_feats = self.model_backbone(batch)
            query_feats.append(batch_feats)
        query_feats = torch.cat(query_feats)
        query_feats = F.normalize(query_feats, p=2, dim=1)
        return query_feats

    def draw_found_masks(img, roi, mask):
        return img
