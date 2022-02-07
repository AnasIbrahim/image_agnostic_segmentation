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

def find_object_mask(img_original, object_images_path, predictions):
    object_images_paths = glob.glob(object_images_path+'/*')

    features_all = list()

    sift = cv2.SIFT_create()
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher()

    instances = predictions["instances"].to("cpu")

    # Visualize masks
    for i in range(len(instances)):
        # TODO should bounding boxex be used instead of masks
        # mask original image with predicted mask
        instance = instances[i]
        # masks = np.array([instances.pred_masks[i].cpu().detach().numpy() for i in range(len(instance))])
        mask = instance.pred_masks.cpu().detach().numpy()[0]
        # pred_masks = instance.pred_masks[index].cpu().detach().numpy()
        masked_img = img_original.copy()
        masked_img[mask == False] = np.array([0, 0, 0])

        cv2.imshow('Masked_image', masked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img = img_original.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(img, None)

    for image_path in object_images_paths:
        object_img = cv2.imread(image_path)
        object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

        features_per_face = list()

        # find the keypoints and descriptors with ORB
        kp2, des2 = sift.detectAndCompute(object_img, None)

        # Match descriptors.
        matches = bf.knnMatch(des1, des2, k=2)

        # Sort them in the order of their distance.
        #matches = sorted(matches, key=lambda x: x.distance)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        # Draw first 40 matches.
        img_matches = np.empty((max(img.shape[0], object_img.shape[0]), img.shape[1] + object_img.shape[1], 3),
                               dtype=np.uint8)
        #img3 = cv2.drawMatches(img, kp1, object_img, kp2, matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3 = cv2.drawMatchesKnn(img, kp1, object_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('Good Matches', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        list_kp1 = [kp1[mat[0].queryIdx].pt for mat in good_matches]
        list_kp1 = [(int(element[0]), int(element[1])) for element in list_kp1]

        for i in range(len(instances)):
            # TODO should bounding boxex be used instead of masks
            # mask original image with predicted mask
            instance = instances[i]
            mask = instance.pred_masks.cpu().detach().numpy()[0]
            features_count = 0
            for pixel in list_kp1:
                if mask[pixel[1], pixel[0]] == True:
                    features_count += 1
            #features_count = len(mask[mask[list_kp1] == True])
            features_per_face.append(features_count)

        features_all.append(features_per_face)

    print(features_all)
    features_all = list(map(list, zip(*features_all)))
    print(features_all)

    # normalize number of features
    sum = np.sum(np.array(features_all).flatten())
    features_all = [[float(j)/sum for j in i] for i in features_all]
    print(features_all)

    # TODO use a Threshold for multi mask detection instead of max
    print([np.sum(x) for x in features_all])
    ind = np.argmax([np.sum(x) for x in features_all])

    return predictions[ind]


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
