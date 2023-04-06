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


class UnseenSegment:
    def __init__(self, model_path, device, confidence=0.7):
        # --- detectron2 Config setup ---
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # WAS 3x.y
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
        cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
        cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(cfg)

    def segment_image(self, img):
        predictions = self.predictor(img)
        return predictions


class ZeroShotClassification:
    def __init__(self, device, gallery_images_path=None, gallery_buffered_path=None, method='vit', siamese_model_path=None):
    # TODO Save gallery features as a file that can be reloaded
        self.method = method
        self.device = device
        if method == 'vit':
            self.model_backbone = torchvision.models.vit_b_16(weights='DEFAULT')
            self.model_backbone.to(self.device)
        elif method == 'siamese':
            if siamese_model_path is None:
                print("Path for siamese model is missing. Exiting ...")
                exit()
            self.siamese_model = SiameseNetwork()
            self.siamese_model = self.siamese_model.to(self.device)

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

        if gallery_buffered_path is not None:
            with open(gallery_buffered_path, 'rb') as f:
                gallery_dict = pickle.load(f)
            self.gallery_obj_names = gallery_dict['gallery_obj_names']
            self.gallery_feats = gallery_dict['gallery_feats']
            self.gallery_classes = gallery_dict['gallery_classes']

            self.gallery_feats = self.gallery_feats.to(device=self.device)
        elif gallery_images_path is not None:
            self.update_gallery(gallery_images_path)
        else:
            print("Neither gallery images path nor buffered gallery file are provided. Exiting ...")
            exit()

    def save_gallery(self, path):
        gallery_dict = {}
        gallery_dict['gallery_obj_names'] = self.gallery_obj_names
        gallery_dict['gallery_feats'] = self.gallery_feats
        gallery_dict['gallery_classes'] = self.gallery_classes
        with open(path, 'wb') as f:
            pickle.dump(gallery_dict, f)

    @torch.no_grad()
    def update_gallery(self, gallery_path):
        print("Extracting/updating gallery features")
        # Extract gallery images
        self.gallery_obj_names = os.listdir(gallery_path)
        gallery_images = []
        self.gallery_feats = []
        self.gallery_classes = []
        for obj_num, obj_path in enumerate(self.gallery_obj_names):
            obj_images_path = glob.glob(os.path.join(gallery_path, obj_path, '*'))
            for obj_image_path in obj_images_path:
                obj_image = Image.open(obj_image_path).convert("RGB")
                angles = [0, 45, 90, 135, 180, 225, 270, 315]
                gallery_images += [self.transform(obj_image.rotate(angle)) for angle in angles]
            self.gallery_classes += [obj_num] * len(obj_images_path) * len(angles)  # multiply angles for augmentation
        self.gallery_classes = np.array(self.gallery_classes)
        gallery_images = torch.stack(gallery_images)
        gallery_images = gallery_images.to(device=self.device)
        # split image to fit into memory
        if self.device == 'cuda':
            gallery_images = torch.split(gallery_images, 200)  # TODO use a batch of 200 to fit GPU memory
        else: # self.device == 'cpu'
            gallery_images = (gallery_images)
        self.gallery_feats = []
        for batch in gallery_images:
            if self.method == 'vit':
                batch_feats = self.model_backbone(batch)
            elif self.method == 'siamese':
                batch_feats = self.siamese_model.extract_gallery_feats(batch)
            self.gallery_feats.append(batch_feats)
        self.gallery_feats = torch.cat(self.gallery_feats)
        self.gallery_feats = F.normalize(self.gallery_feats, p=2, dim=1)  # TODO recheck normalization

    def find_object(self, rgb_img, predictions, obj_name):
        query_feats = self.extract_query_feats(rgb_img, predictions)
        obj_feats = self.gallery_feats[np.where(self.gallery_classes == self.gallery_obj_names.index(obj_name))[0]]
        if self.method == 'vit':
            dists = F.cosine_similarity(query_feats.unsqueeze(1), obj_feats, dim=-1)
        elif self.method == 'siamese':
            dists = self.siamese_model.dist_measure(query_feats.repeat(obj_feats.shape[0], 1), obj_feats.repeat(query_feats.shape[0], 1))
        dists = dists.reshape(obj_feats.shape[0], query_feats.shape[0])
        dists = torch.transpose(dists, 0, 1)
        matched_query = torch.argmax(torch.max(dists, dim=1)[0]).item()  # TODO: change method to find several occurrences
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
                if self.method == 'vit':
                    dists = torch.cosine_similarity(query_feat, obj_feats, dim=1)
                elif self.method == 'siamese':
                    dists = self.siamese_model.dist_measure(query_feat.repeat(obj_feats.shape[0], 1), obj_feats)
                obj_dists.append(np.array(dists.cpu()))
            obj_dists = [max(obj) for obj in obj_dists]  # TODO check for a threshold where the object is not in query
            pred_class = obj_dists.index(max(obj_dists))
            pred_classes.append(pred_class)
        classified_predictions['instances'].pred_classes = torch.tensor(pred_classes)
        return classified_predictions
        #classified_predictions = copy.deepcopy(predictions)
        #query_feats = self.extract_query_feats(rgb_img, predictions)
        #if self.method == 'vit':
        #    dists = F.cosine_similarity(query_feats.unsqueeze(1), self.gallery_feats, dim=-1)
        #elif self.method == 'siamese':
        #    dists = self.siamese_model.dist_measure(query_feats.repeat(self.gallery_feats.shape[0], 1), self.gallery_feats.repeat(query_feats.shape[0], 1))
        #dists = dists.reshape(self.gallery_feats.shape[0], query_feats.shape[0])
        #dists = torch.transpose(dists, 0, 1)
        ## TODO check for a threshold to check if object is is not in query to prevent classification of objects not in gallery
        #pred_classes = torch.tensor(self.gallery_classes[torch.argmax(dists,dim=1).tolist()])
        #classified_predictions['instances'].pred_classes = pred_classes
        #return classified_predictions

    @torch.no_grad()
    def extract_query_feats(self, rgb_img, predictions):
        query_images = []
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
            query_images.append(query_img)
        query_images = torch.stack(query_images)
        query_images = query_images.to(device=self.device)
        if self.method == 'vit':
            query_feats = self.model_backbone(query_images)
        elif self.method == 'siamese':
            query_feats = self.siamese_model.extract_query_feats(query_images)
        query_feats = F.normalize(query_feats, p=2, dim=1)  # TODO recheck normalization
        return query_feats

    def draw_found_masks(img, roi, mask):
        return img
