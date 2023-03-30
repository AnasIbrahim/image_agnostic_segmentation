import copy
import glob
import os
import numpy as np
import cv2
import torch
from PIL import Image
from collections import OrderedDict

import torchvision
from torchvision import transforms

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import coco_evaluation

from .siamese_network import SiameseNetwork

setup_logger()  # initialize the detectron2 logger and set its verbosity level to “DEBUG”.


def segment_image(img, model_path):
    confidence = 0.7

    # --- detectron2 Config setup ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # WAS 3x.y
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    predictions = predictor(img)

    return predictions


def draw_segmented_image(img, predictions, classes=['']):
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


class DoUnseen:
    def __init__(self, gallery_path, method='vit', siamese_model_path=None):
    # TODO Save gallery features as a file that can be reloaded
        self.method = method
        if method == 'vit':
            self.model_backbone = torchvision.models.vit_b_16(weights='DEFAULT')
        elif method == 'siamese':
            if siamese_model_path is None:
                print("Path for siamese model is missing. Exiting ...")
                exit()
            self.siamese_model = SiameseNetwork()

            checkpoint = torch.load(siamese_model_path, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']

            model_dict = self.siamese_model.state_dict()
            new_state_dict = OrderedDict()
            matched_layers, discarded_layers = [], []

            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:] # discard module.

                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
                    matched_layers.append(k)
                else:
                    discarded_layers.append(k)

            model_dict.update(new_state_dict)
            self.siamese_model.load_state_dict(model_dict)

        self.feed_shape = [3, 224, 224]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(self.feed_shape[1:], antialias=True)
        ])

        self.update_gallery(gallery_path)

    @torch.no_grad()
    def update_gallery(self, gallery_path):
        # Extract gallery images
        self.gallery_feats = {}
        obj_list = os.listdir(gallery_path)
        for obj_num, obj_path in enumerate(obj_list):
            obj_images = [self.transform(Image.open(path).convert("RGB")) for path in glob.glob(os.path.join(gallery_path, obj_path, '*'))]
            obj_images = torch.stack(obj_images)
            # TODO use bigger batch size to save time
            if self.method == 'vit':
                obj_feats = [self.model_backbone(obj_images)]
            elif self.method == 'siamese':
                obj_feats = [self.siamese_model.extract_gallery_feats(obj_images)]
            self.gallery_feats[obj_list[obj_num]] = torch.stack(obj_feats).squeeze(dim=0)

    def find_object(self, rgb_img, predictions, obj_name):
        query_feats = self.extract_query_feats(rgb_img, predictions)
        obj_feats = self.gallery_feats[obj_name]
        obj_dists = []
        for query_id, query_feat in enumerate(query_feats):
            if self.method == 'vit':
                dists = torch.cosine_similarity(query_feat, obj_feats, dim=1)
            elif self.method == 'siamese':
                dists = self.siamese_model.dist_measure(query_feat.repeat(obj_feats.shape[0], 1), obj_feats)
            obj_dists.append(np.array(dists))
        obj_dists = [max(obj) for obj in obj_dists]
        matched_query = obj_dists.index(max(obj_dists))  # TODO: change method to find several occurrences
        obj_predictions = copy.deepcopy(predictions)
        obj_predictions['instances'] = obj_predictions['instances'][matched_query]

        return obj_predictions

    def classify_all_objects(self, rgb_img, predictions):
        classified_predictions = copy.deepcopy(predictions)
        pred_classes = []
        query_feats = self.extract_query_feats(rgb_img, predictions)
        for query_id, query_feat in enumerate(query_feats):
            obj_dists = []
            for obj_num, obj_name in enumerate(self.gallery_feats.keys()):
                obj_feats = self.gallery_feats[obj_name]
                if self.method == 'vit':
                    dists = torch.cosine_similarity(query_feat, obj_feats, dim=1)
                elif self.method == 'siamese':
                    dists = self.siamese_model.dist_measure(query_feat.repeat(obj_feats.shape[0], 1), obj_feats)
                obj_dists.append(np.array(dists))
            obj_dists = [max(obj) for obj in obj_dists]  # TODO check for a threshold where the object is not in query
            pred_class = obj_dists.index(max(obj_dists))
            pred_classes.append(pred_class)
        classified_predictions['instances'].pred_classes = torch.tensor(pred_classes)

        return classified_predictions

    @torch.no_grad()
    def extract_query_feats(self, rgb_img, predictions):
        query_images_feats = []
        instances = predictions['instances'].to('cpu')
        for idx in range(len(instances)):
            bbox = instances[idx].pred_boxes.tensor.squeeze().numpy()
            mask = instances[idx].pred_masks.squeeze().numpy().astype(np.uint8)
            masked_rgb = cv2.bitwise_or(rgb_img, rgb_img, mask=mask)
            bbox = [int(val) for val in bbox]
            obj_cropped_mask = masked_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            #cv2.imshow('image', obj_cropped_mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            query_img = self.transform(cv2.cvtColor(obj_cropped_mask, cv2.COLOR_BGR2RGB))
            query_img = torch.unsqueeze(query_img, dim=0)
            if self.method == 'vit':
                feat = self.model_backbone(query_img)
            elif self.method == 'siamese':
                feat = self.siamese_model.extract_query_feats(query_img)
            query_images_feats.append(feat)
        return query_images_feats

    def draw_found_masks(img, roi, mask):
        return img
