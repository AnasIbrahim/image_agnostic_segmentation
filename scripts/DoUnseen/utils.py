import copy

import numpy as np


def merge_masks(original_masks, rgb_color_image):
    # convert the input list of masks to a numpy array
    masks = [original_masks[i]['segmentation'] for i in range(len(original_masks))]
    masks = copy.deepcopy(np.array(masks))

    # initialize a list to store the merged masks
    merged_masks = []
    merged_masks_binary = []

    # loop over each mask in the input list
    for i in range(len(masks)):
        #plot_mask([masks[i]], rgb_color_image)
        # check if the current mask is already included in a merged mask
        already_merged = False
        for j in range(len(merged_masks)):
            if np.array_equal(masks[i], merged_masks[j]):
                already_merged = True
                break

        if already_merged:
            continue

        # find the overlap between the current mask and all the other masks
        overlap = np.zeros_like(masks[i])
        for j in range(len(masks)):
            #plot_mask([masks[j]], rgb_color_image)
            if i == j:
                continue

            overlap += np.logical_and(masks[i], masks[j])

        # check if more than 90% of the current mask is inside another mask
        if np.sum(overlap) / np.sum(masks[i]) > 0.9:
             #merge the current mask with the overlapping mask(s)
            merged_mask = masks[i]
            for j in range(len(masks)):
                if i == j:
                    continue

                if np.sum(np.logical_and(masks[i], masks[j])) > 0:
                    merged_mask = np.logical_or(merged_mask, masks[j])
            # add the merged mask to the list of merged masks
            merged_masks_binary.append(merged_mask)

            new_mask = original_masks[j].copy()
            # replace the segmentation mask with the merged mask
            new_mask['segmentation'] = merged_mask
            # get the area of the merged mask
            new_mask['area'] = np.sum(merged_mask)
            # calculate the new bounding box of the merged mask
            new_mask['bbox'] = [np.min(np.where(merged_mask)[1]), np.min(np.where(merged_mask)[0]), np.max(np.where(merged_mask)[1]), np.max(np.where(merged_mask)[0])]
            merged_masks.append(new_mask)
        else:
            merged_masks_binary.append(masks[i])
            # add the current mask to the list of merged masks
            merged_masks.append(original_masks[i])

    return merged_masks
