import itertools
import os
import random
from imgaug.augmenters.weather import FastSnowyLandscape

import numpy as np
import six
from cv2 import cv2

from augmentation import seq


class DataLoaderError(Exception):
    pass


def get_list_for_data(data_list_path):
    """将划分好的数据路径与标签读入成一个列表"""
    data_list = []
    with open(data_list_path, 'r') as f_data:
        lines = f_data.readlines()
    for line in lines:
        file_path_01 = line.strip().split(',')[0].strip()
        file_path_02 = line.strip().split(',')[1].strip()
        label = line.strip().split(',')[2].strip()
        if os.path.exists(file_path_01) and os.path.exists(file_path_02):
            # if resize_pad_check(file_path_01, 64, 160) and resize_pad_check(file_path_02, 64, 160):
            data_list.append((file_path_01, file_path_02, label))
        else:
            pass
        # print(file_path_01, file_path_02, label)
    return data_list


def get_image_array(image_input,
                    height, width,
                    is_training=True,
                    do_augment=False):
    if type(image_input) is np.ndarray:
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.exists(image_input):
            raise DataLoaderError(
                "get_image_array: path '{}' doesn't exist".format(image_input))
        # img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(image_input, cv2.IMREAD_COLOR)

    if is_training:
        if do_augment:
            asdzz = random.randint(0, 9)
            if asdzz > 8:
                # 自适应阈值二值化
                img_aug = erzhi(img)
                img_aug = np.expand_dims(img_aug, axis=2)
                img_aug = np.concatenate((img_aug, img_aug, img_aug), axis=-1)
                img_aug = seq.augment_image(img_aug)
            else:
                img_aug = seq.augment_image(img)
        else:
            img_aug = img
    else:
        img_aug = img

    # img_aug = resize_pad_old(img_aug, height, width)
    img_aug = resize_pad(img_aug, height, width)

    return img_aug


def data_generate(data_list_path,
                  batch_size,
                  height, width,
                  is_training=True,
                  do_augment=False):
    if is_training:
        data_list_path = os.path.join(data_list_path, 'train_data.csv')
    else:
        data_list_path = os.path.join(data_list_path, 'test_data.csv')

    data_list = get_list_for_data(data_list_path)
    if is_training:
        print('**********len_train_data:', len(data_list))

    random.shuffle(data_list)
    zipped = itertools.cycle(data_list)

    while True:
        img_01_list = []
        img_02_list = []
        label_list_1 = []
        label_list_2 = []
        label_list_3 = []
        for _ in range(batch_size):
            img_path_01, img_path_02, label = next(zipped)

            img_01 = get_image_array(img_path_01,
                                     height=height,
                                     width=width,
                                     is_training=is_training,
                                     do_augment=do_augment)
            img_02 = get_image_array(img_path_02,
                                     height=height,
                                     width=width,
                                     is_training=is_training,
                                     do_augment=do_augment)
            # img_merge = np.concatenate((img_01, img_02), axis=2)
            # img_list.append(img_merge)
            img_01_list.append(img_01)
            img_02_list.append(img_02)
            label_list_1.append(label)
            label_list_2.append(label)
            label_list_3.append(label)
            # print(img_path_01)
            # print(img_path_02)

        yield [np.array(img_01_list), np.array(img_02_list)], [np.array(label_list_1), np.array(label_list_2), np.array(label_list_3)]


def resize_pad_old(im, height, width):
    try:
        channel = im.shape[2]
    except Exception as e:
        print(e)
    # resize
    hei_ori, wid_ori = im.shape[:2]
    a1 = width/wid_ori
    a2 = height/hei_ori
    scale = min(a1, a2)
    im_resize = cv2.resize(im,
                           (round(wid_ori * scale), round(hei_ori * scale)),
                           interpolation=cv2.INTER_AREA)

    # pad   可修改随机填充
    h, w = im_resize.shape[:2]
    top_pad = (height - h) // 2
    bottom_pad = height - h - top_pad
    left_pad = (width - w) // 2
    right_pad = width - w - left_pad
    if channel == 1:
        padding = [(top_pad, bottom_pad), (left_pad, right_pad)]  # 单通道图
    elif channel == 3:
        padding = [(top_pad, bottom_pad),
                   (left_pad, right_pad), (0, 0)]  # 三通道图
    image = np.pad(im_resize, padding, mode='constant', constant_values=255)

    return image


def resize_pad(im, height, width):
    try:
        channel = im.shape[2]
    except Exception as e:
        print(e)
    # resize
    hei_ori, wid_ori = im.shape[:2]
    scale = height/hei_ori
    if scale > 1.0:
        im_resize = cv2.resize(im,
                               (round(wid_ori * scale), round(hei_ori * scale)),
                               interpolation=cv2.INTER_CUBIC)  # 三次样条插值
    elif scale <= 1.0:
        im_resize = cv2.resize(im,
                               (round(wid_ori * scale), round(hei_ori * scale)),
                               interpolation=cv2.INTER_AREA)  # 区域插值

    # pad   可修改随机填充
    h, w = im_resize.shape[:2]
    top_pad = (height - h) // 2
    bottom_pad = height - h - top_pad
    left_pad = (width - w) // 2
    right_pad = width - w - left_pad

    if channel == 1:
        padding = [(top_pad, bottom_pad), (left_pad, right_pad)]  # 单通道图
    elif channel == 3:
        padding = [(top_pad, bottom_pad),
                   (left_pad, right_pad), (0, 0)]  # 三通道图
    if left_pad >= 0 and right_pad >= 0:
        image = np.pad(im_resize, padding, mode='constant',
                       constant_values=255)
    else:
        # print(padding)
        image = im_resize[0:h, 0-left_pad:w+right_pad]
    return image


def erzhi(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_res = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
    return img_res


if __name__ == "__main__":
    pass
    file_path = '/home/data/ETOP/project_IDN/data'
    save_path = '/home/data/ETOP/project_IDN/check_data'

    # # ================================================================================
    # # file_path = '/home/mortal/桌面/data/ETOP/project_IDN/data/train_data.csv'
    # # file_path = '/home/mortal/桌面/data/ETOP/project_IDN/data/val_data.csv'
    # file_path = '/home/mortal/桌面/data/ETOP/project_IDN/data/test_data.csv'
    # lis_data = get_list_for_data(file_path)
    # print(len(lis_data))
    # # ================================================================================

    from config import *
    check_data_gen = data_generate(data_list_path=file_path,
                                   batch_size=1,
                                   height=INPUT_HEIGHT,
                                   width=INPUT_WIDTH,
                                   is_training=True,
                                   do_augment=True)
    a, b = 1, 1
    for lis_img, lis_label in check_data_gen:
        label = lis_label[0].tolist()
        label = label[0]
        label_text = 'label:{}'.format(label)

        img_01 = lis_img[0]
        img_02 = lis_img[1]
        # print('img_01 OLD:', img_01.shape)
        # print('img_02 OLD:', img_02.shape)
        img_01 = img_01.reshape(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL)
        img_02 = img_02.reshape(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL)

        img_vmerge = np.vstack((img_01, img_02))
        cv2.putText(img_vmerge, label_text, (100, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 118, 72), 2)
        cv2.imshow('check', img_vmerge)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            exit()

        # # print('img_01:', img_01.shape)
        # # print('img_02:', img_02.shape)
        # img_01_name = 'img_01_{}.png'.format(a)
        # img_02_name = 'img_02_{}.png'.format(b)
        # img_01_save = os.path.join(save_path, img_01_name)
        # img_02_save = os.path.join(save_path, img_02_name)
        # cv2.imwrite(img_01_save, img_01)
        # cv2.imwrite(img_02_save, img_02)
        # print(img_01_save, '--save--')
        # print(img_02_save, '--save--')
        # a += 1
        # b += 1

    # img_path = '/home/mortal/桌面/data/Datasets/ETOP_HWDS_data/full_forg/forgeries_1_2.png'
    # img = cv2.imread(img_path)
    # print(img.shape)
    # img_re = erzhi(img)
    # print(img_re.shape)
    # img_aug = np.expand_dims(img_re, axis=2)
    # img_aug = np.concatenate((img_aug, img_aug, img_aug), axis=-1)
    # print(img_aug.shape)
    # cv2.imwrite('/home/mortal/桌面/data/ETOP/project_IDN/asdzz/asdzz.png',img_re)
