import numpy as np

import cv2
import copy


def merge_masks(original_masks, rgb_color_image):
    '''
    Remove masks that are inside other masks (> 0.9).
    Masks must be ordered from smaller to bigger.
    '''
    # convert the input list of masks to a numpy array
    binary_masks = [original_masks[i]['segmentation'] for i in range(len(original_masks))]
    parent_masks = []

    # loop over the masks, if a mask if not inside another mask, add it to the list of parent masks
    for mask_id, binary_mask in enumerate(binary_masks):
        # loop over the masks again to check if the current mask is inside another mask
        is_inside = False
        for mask_id2, binary_masks2 in enumerate(binary_masks[mask_id+1:]):
            # if more than 0.9 of the current mask is inside another mask, break the loop
            interection = np.logical_and(binary_mask, binary_masks2)
            if np.sum(interection) / np.sum(binary_mask) > 0.9:
                is_inside = True
                break
        # if the current mask is not inside another mask, add it to the list of parent masks
        if not is_inside:
            parent_masks.append(original_masks[mask_id])

    return parent_masks
