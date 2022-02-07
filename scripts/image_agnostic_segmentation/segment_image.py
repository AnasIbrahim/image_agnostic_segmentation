#!/usr/bin/env python
import os

import cv2
import argparse

import agnostic_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="path to image", required=True)
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model", required=True)
    parser.add_argument("--objects-images", type=str, help="path to object images to be segmented \
                                                            ex: objects_folder/[book,shampoo,...]")
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    predictions = agnostic_segmentation.segment_image(img, args.model_path)
    seg_img = agnostic_segmentation.draw_segmented_image(img, predictions)

    cv2.imshow('segmented_image', seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.objects_images_folder is not None:
        objects = os.listdir(args.objects_images_folder)
        for object in objects:
            object_predictions = agnostic_segmentation.find_object_mask(img, os.path.join(args.objects_images_folder,object), predictions)
            agnostic_segmentation.draw_segmented_image(args.img, object_predictions)

            cv2.imshow(object + '  masks', seg_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
