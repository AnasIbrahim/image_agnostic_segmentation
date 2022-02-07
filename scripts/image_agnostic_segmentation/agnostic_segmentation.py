import glob

import numpy as np
import cv2

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


setup_logger()  # initialize the detectron2 logger and set its verbosity level to “DEBUG”.


def segment_image(img, model_path):
    confidence = 0.9

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
    MetadataCatalog.get("user_data").set(thing_classes=["object"])
    metadata = MetadataCatalog.get("user_data")
    v = Visualizer(img,
                   metadata=metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    img = out.get_image()
    
    return img

def find_object_mask(img, object_images_path, predictions):
    object_images_paths = glob.glob(object_images_path+'/*')

    features_per_mask = list()  # list of list of features per mask of features per object face

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    instances = predictions["instances"].to("cpu")
    for i in range(len(instances)):
        # TODO should bounding boxex be used instead of masks
        # mask original image with predicted mask
        instance = instances[i]
        #masks = np.array([instances.pred_masks[i].cpu().detach().numpy() for i in range(len(instance))])
        mask = instance.pred_masks.cpu().detach().numpy()[0]
        #pred_masks = instance.pred_masks[index].cpu().detach().numpy()
        masked_img = img.copy()
        masked_img[mask == False] = np.array([0, 0, 0])

        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Masked_image', masked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        kp1 = orb.detect(masked_img, None)
        kp1, des1 = orb.compute(masked_img, kp1)

        features_per_object_face = list()
        for image_path in object_images_paths:
            object_img = cv2.imread(image_path)

            object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

            # find the keypoints and descriptors with ORB
            kp2 = orb.detect(object_img, None)
            kp2, des2 = orb.compute(object_img, kp2)

            # Match descriptors.
            matches = bf.match(des1, des2)

            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            features_per_object_face.append(len(matches))

            # Draw first 40 matches.
            #img_matches = np.empty((max(img.shape[0], object_img.shape[0]), img.shape[1] + object_img.shape[1], 3), dtype=np.uint8)
            #img3 = cv2.drawMatches(masked_img, kp1, object_img, kp2, matches[:40], img_matches,
            #               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            #cv2.imshow('Good Matches', img3)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        features_per_mask.append(features_per_object_face)

    print(features_per_mask)
    print([np.sum(x) for x in features_per_mask])

def draw_found_masks(img, object_predictions, object):
    MetadataCatalog.get("user_data").set(thing_classes=[object])
    metadata = MetadataCatalog.get("user_data")
    v = Visualizer(img,
                   metadata=metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(object_predictions["instances"].to("cpu"))
    img = out.get_image()

    return img
