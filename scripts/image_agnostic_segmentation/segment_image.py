#!/usr/bin/env python
import os

import cv2
import argparse

import agnostic_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="path to image", default='../../demo/test_1.png')
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model",
                        default='../../models/FAT_trained_Ml2R_bin_fine_tuned.pth')
    parser.add_argument("--objects-images-folder", type=str,
                        help="path to object images to be segmented, ex: objects_folder/[book,shampoo,...]",
                        default='../../demo/objects')
    args = parser.parse_args()

    # get absolute path
    args.image_path = os.path.abspath(args.image_path)
    args.model_path = os.path.abspath(args.model_path)
    args.objects_images_folder = os.path.abspath(args.objects_images_folder)

    img = cv2.imread(args.image_path)
    predictions = agnostic_segmentation.segment_image(img, args.model_path)
    seg_img = agnostic_segmentation.draw_segmented_image(img, predictions)

    cv2.imshow('segmented_image', seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.objects_images_folder is not None:
        objects = os.listdir(args.objects_images_folder)
        for obj in objects:
            print('--------------------------------')
            print('Searching for object: ', obj)
            object_instances = agnostic_segmentation.find_object_mask(img, os.path.join(args.objects_images_folder,obj), predictions)
            if len(object_instances) == 0:
                print('No matches found for object: ', obj)
            else:
                print('Found ', str(len(object_instances)), ' matches for object ', obj)
                found_masks_img = agnostic_segmentation.draw_found_masks(img, object_instances, obj)

                cv2.imshow(obj + '  masks', found_masks_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
