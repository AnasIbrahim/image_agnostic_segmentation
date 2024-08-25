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


def pose_from_vector3D(waypoint):
    """
    calculates a quaternion from a 3D vector (half-way vector)
    https://answers.ros.org/question/228896/quaternion-of-a-3d-vector/
    http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    """
    x = waypoint[0]
    y = waypoint[1]
    z = waypoint[2]
    #calculating the half-way vector.
    u = [1,0,0]
    norm = np.linalg.norm(waypoint)
    v = np.asarray(waypoint)/norm
    if (np.array_equal(u, v)):
        w = 1
        x = 0
        y = 0
        z = 0
    elif (np.array_equal(u, np.negative(v))):
        w = 0
        x = 0
        y = 0
        z = 1
    else:
        half = [u[0]+v[0], u[1]+v[1], u[2]+v[2]]
        w = np.dot(u, half)
        temp = np.cross(u, half)
        x = temp[0]
        y = temp[1]
        z = temp[2]
    norm = np.math.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0:
        norm = 1
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array([x, y, z, w])

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()