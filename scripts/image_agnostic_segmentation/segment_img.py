#!/usr/bin/env python
import cv2
import argparse

from agnostic_segmentation_live_demo import agnostic_segmentation

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--img", type=str, help="path to image", default="/home/gouda/segmentation/test_images/4.png")
    parser.add_argument("--img", type=str, help="path to image", default="/home/gouda/segmentation/test_images/10.png")
    #parser.add_argument("--img", type=str, help="path to image", default="/media/gouda/all/datasets/HOPE/hope_val/val/000001/rgb/000000.png")
    #parser.add_argument("--model_path", type=str, help="path to class-agnostic model", default='/home/gouda/segmentation/seg_ws/src/klt_dataset_collector/agnostic_segmentation_live_demo/agnostic_segmentation_model.pth')
    parser.add_argument("--model_path", type=str, help="path to class-agnostic model", default='/media/gouda/all/sciebo/bin_picking/fine_tuning/fine_tuning_training_output/model_final.pth')
    args = parser.parse_args()
    img = cv2.imread(args.img)
    seg_img = agnostic_segmentation.segment_image(img, img, args.model_path)
    cv2.imshow('Image', seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(args.img[:-4]+'_segmentated.png',seg_img)


if __name__ == '__main__':
    main()
