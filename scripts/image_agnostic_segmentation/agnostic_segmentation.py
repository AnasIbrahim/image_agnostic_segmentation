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
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    img = out.get_image()
    
    return img

def find_object_mask(img_original, object_images_path, predictions):
    object_images_paths = glob.glob(object_images_path+'/*')

    Threshold = 30

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    instances = predictions["instances"].to("cpu")

    mask_annotations = list()

    for i in range(len(instances)):
        # TODO should bounding boxex be used instead of masks
        instance = instances[i]
        mask = instance.pred_masks.cpu().detach().numpy()[0]
        masked_img = img_original.copy()
        masked_img[mask == False] = np.array([0, 0, 0])

        #cv2.imshow('Masked_image', masked_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(masked_img, None)

        features_per_face = list()
        for image_path in object_images_paths:
            object_img = cv2.imread(image_path)
            object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)


            kp2, des2 = sift.detectAndCompute(object_img, None)

            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_matches.append([m])

            good_matches = sorted([item[0] for item in good_matches], key=lambda x: x.distance)
            good_matches = [[item] for item in good_matches]

            # Draw matches
            #img3 = cv2.drawMatchesKnn(masked_img, kp1, object_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #cv2.imshow('Good Matches', img3)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            list_kp1 = [kp1[mat[0].queryIdx].pt for mat in good_matches]
            #list_kp1 = [(int(element[0]), int(element[1])) for element in list_kp1]

            features_per_face.append(len(list_kp1))

        print(features_per_face)

        if np.max(features_per_face) > Threshold:
            mask_annotations.append(i)

    print(mask_annotations)
    return instances[mask_annotations]


def draw_found_masks(img, object_instances, object):
    MetadataCatalog.get(object+"_data").set(thing_classes=[object])
    metadata = MetadataCatalog.get(object+"_data")
    v = Visualizer(img,
                   metadata=metadata,
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(object_instances)
    img = out.get_image()

    return img
