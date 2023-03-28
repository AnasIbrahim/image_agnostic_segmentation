import glob
import os
import numpy as np
import cv2
from PIL import Image

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

from pycocotools.coco import COCO
from pycocotools.coco import maskUtils


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


def draw_segmented_image(img, predictions):
    MetadataCatalog.get("user_data").set(thing_classes=[""])
    metadata = MetadataCatalog.get("user_data")
    v = Visualizer(img,
                   metadata=metadata,
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    img = out.get_image()

    return img


class DoUnseen:
    def __init__(self, gallery_path, method='vit'):
    # TODO Save gallery features as a file that can be reloaded
        if method == 'vit':
            self.model_backbone = torchvision.models.vit_b_16()
        elif method == 'siamese':
            raise NotImplementedError("Not Implemented")
            #self.siamese_model = siamese()
            #self.model_backbone = siamese_model.backbone

        self.feed_shape = [3, 224, 224]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(self.feed_shape[1:])
        ])

        self.update_gallery(gallery_path)

    def update_gallery(self, gallery_path):
        # Extract gallery images
        self.gallery_feats = {}
        obj_list = os.listdir(gallery_path)
        for obj_num, obj_path in enumerate(obj_list):
            obj_images = [self.transform(Image.open(path).convert("RGB")) for path in glob.glob(os.path.join(gallery_path, obj_path, '*'))]
            self.gallery_feats_[obj_list[obj_num]] = obj_images

    def find_object(self, img_original, predictions):
        query_feats = self.extract_query_feats(img_original, predictions)
        return object_prediction

    def classify_all_objects(self, rgb_img, predictions):
        query_feats = self.extract_query_feats(rgb_img, predictions)
        return classified_predictions

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
            query_images_feats.append(self.transform(cv2.cvtColor(obj_cropped_mask, cv2.COLOR_BGR2RGB)))

    def draw_found_masks(img, roi, mask):
        return img
