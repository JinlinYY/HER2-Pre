
import numpy as np
import cv2
import os
import json
from labelme import utils


def create_file(file):
    if not os.path.exists(file):
        os.mkdir(file)


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()



def from_mask_extract_ROI(img, mask):
    y_indices, x_indices = np.where(mask == 1)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return None

    min_y, max_y = np.min(y_indices), np.max(y_indices)
    min_x, max_x = np.min(x_indices), np.max(x_indices)

    # Clip coordinates to image boundaries
    min_y = max(0, min_y)
    max_y = min(img.shape[0] - 1, max_y)
    min_x = max(0, min_x)
    max_x = min(img.shape[1] - 1, max_x)

    if min_y > max_y or min_x > max_x:
        return None  # Invalid ROI

    mask_ROI = mask[min_y:max_y + 1, min_x:max_x + 1]
    ROI = img[min_y:max_y + 1, min_x:max_x + 1]


    mask = np.ones_like(ROI)
    if ROI.ndim == 3:
        for i in range(mask.shape[2]):
            mask[:, :, i] = mask_ROI
    else:
        mask = mask_ROI

    masked_ROI = mask * ROI


    if masked_ROI.ndim == 3:
        channel_reverse = np.ones_like(masked_ROI)
        for i in range(channel_reverse.shape[2]):
            channel_reverse[:, :, -1 - i] = masked_ROI[:, :, i]
    else:
        channel_reverse = masked_ROI

    return channel_reverse

def from_json_extract_ROI_based_label(file):
    file_list = os.listdir(file)

    json_file = []
    for idx, value in enumerate(file_list):
        if value.split('.')[-1] == 'json':
            json_name = os.path.join(file, value)
            json_file.append(json_name)

    for idx, value in enumerate(json_file):
        json_name = value
        with open(json_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img = utils.img_b64_to_arr(data['imageData'])
        lbl, lbl_name = utils.labelme_shapes_to_label(img.shape, data['shapes'])

        mask = []
        for i in range(1, len(lbl_name)):
            mask.append((lbl == i).astype(np.uint8))

        if not mask:
            continue

        mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])

        keys = []
        values = []
        for value, key in enumerate(lbl_name):
            key = key.split('-')[0]
            keys.append(key)
            values.append(value)
        labels = []
        for i in range(1, len(values)):
            labels.append(keys[i])

        for i in range(mask.shape[2]):
            a = mask[:, :, i]
            masked_ROI_ChannelRevrse = from_mask_extract_ROI(img=img, mask=a)

            if masked_ROI_ChannelRevrse is None:
                continue

            save_file = str(labels[i])
            create_file(save_file)
            save_ROI = './' + save_file + '/' + json_name.split('/')[-1].split('.json')[0] + '-' + str(i) + '.jpg'
            cv2.imwrite(save_ROI, masked_ROI_ChannelRevrse)




from_json_extract_ROI_based_label(file='HER2-16/HER2_zero-16/HER2_zero_breast-16/')