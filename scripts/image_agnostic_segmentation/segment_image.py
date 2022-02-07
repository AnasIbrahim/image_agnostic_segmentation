#!/usr/bin/env python
import cv2
import argparse

import agnostic_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="path to image", required=True)
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model", required=True)
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    predictions = agnostic_segmentation.segment_image(img, args.model_path)
    seg_img = agnostic_segmentation.visualize_segmented_img(img, predictions)

    cv2.imshow('segmented_image', seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
