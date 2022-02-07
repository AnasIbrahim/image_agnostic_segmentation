import numpy as np

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


setup_logger()  # initialize the detectron2 logger and set its verbosity level to “DEBUG”.


def segment_image(img, model_path):
    confidence = 0.95

    # --- var dec ---

    # --- Config setup ---
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


def visualize_segmented_img(img, predictions):
    MetadataCatalog.get("user_data").set(thing_classes=["object"])
    kitchen0_metadata = MetadataCatalog.get("kitchen0_val")
    v = Visualizer(img,
                   metadata=kitchen0_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    img = out.get_image()
    
    return img
