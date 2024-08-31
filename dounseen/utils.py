import cv2
import numpy as np
import copy


def draw_segmented_image(rgb_img, masks, bboxes, classes_predictions=None, classes_names=None):
    """
    draws a list of binary masks over an image.
    If predicted classes are provided, the masks are colored according to the class.
    """
    img = copy.deepcopy(rgb_img)
    opacity = 0.5
    for idx in range(len(masks)):
        mask = masks[idx]
        bbox = bboxes[idx]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # add mask to image with opacity
        img[mask] = ((1 - opacity) * img[mask] + opacity * np.array(color)).astype(np.uint8)
        # add a black border around the mask (using find contours and draw contours)
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 2)
        # add bbox (unfilled) to image
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    # add text to image in a separate loop to make text more visible
    for idx in range(len(bboxes)):
        bbox = bboxes[idx]
        if classes_predictions is not None:
            class_id = classes_predictions[idx]
            if classes_names is not None:
                class_name = classes_names[class_id]
            else:
                class_name = str(class_id)
            # add a blacked filled rectangle of the top left corner of the bbox that would fit the class name text
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + len(class_name) * 12, bbox[1] + 20), (0, 0, 0), -1)
            # add class name inside the rectangle without opacity
            cv2.putText(img, class_name, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def resize_and_pad(img, size, pad_color):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA

    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w) / h
    saspect = float(sw) / sh

    if (saspect >= aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img


def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_image_segments_from_binary_masks(rgb_image, masks, bboxes):
    segments = []
    for idx in range(len(masks)):
        mask = masks[idx]
        bbox = bboxes[idx]
        # add mask to the image with background being white
        segment = np.ones(rgb_image.shape, dtype=np.uint8) * 255
        segment[mask] = rgb_image[mask]
        segment = segment[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        segments.append(segment)
    return segments


def reformat_sam2_output(sam2_output):
    masks = []
    bboxes = []
    for idx in range(len(sam2_output)):
        mask = sam2_output[idx]['segmentation']
        bbox = sam2_output[idx]['bbox']
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # change bboxes from xywh to xyxy
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        masks.append(mask)
        bboxes.append(bbox)
    return masks, bboxes


def remove_unmatched_query_segments(class_predictions, masks, bboxes):
    # remove class predictions with class -1 and their corresponding segments
    new_seg_predictions = []
    new_class_predictions = []
    for idx in range(len(class_predictions)):
        if class_predictions[idx] != -1:
            new_seg_predictions.append(idx)
            new_class_predictions.append(class_predictions[idx])
    masks = [mask for i, mask in enumerate(masks) if i in new_seg_predictions]
    bboxes = [bbox for i, bbox in enumerate(bboxes) if i in new_seg_predictions]
    return new_class_predictions, masks, bboxes
