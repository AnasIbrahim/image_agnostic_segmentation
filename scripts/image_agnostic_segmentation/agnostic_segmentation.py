import cv2
import time
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

import rospkg

setup_logger()  # initialize the detectron2 logger and set its verbosity level to “DEBUG”.

def segment_image(im_input_input, im_input_visualize, model_path):
    confidence = 0.95

    # --- var dec ---
    MetadataCatalog.get("kitchen0_val").set(thing_classes=["object"])
    kitchen0_metadata = MetadataCatalog.get("kitchen0_val")

    # --- Config setup ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # WAS 3x.y
    rospack = rospkg.RosPack()
    rospack.list()
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    start = time.time()
    outputs = predictor(im_input_input)
    end = time.time()
    print("Inference time: ", end - start)
    v = Visualizer(im_input_visualize,
                   metadata=kitchen0_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = out.get_image()

    return img