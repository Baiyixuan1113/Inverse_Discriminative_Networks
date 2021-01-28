import imgaug
from imgaug import augmenters as iaa
from imgaug import parameters as iap


def sometimes(aug):
    return iaa.Sometimes(0.3, aug)


seq = iaa.Sequential(
    iaa.Sometimes(0.3, [
        iaa.SomeOf((0, 3), [
            sometimes(iaa.Affine(
                scale=iap.Uniform(0.5, 0.8),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-5, 5),  # 旋转±3度之间
                shear=(-3, 3),  # 剪切变换±3度（矩形变平行四边形）
                cval=255,  # 规定填充的像素值
            )),
            sometimes(iaa.OneOf([
                iaa.GaussianBlur((0, 0.8)),  # 高斯模糊
                iaa.AverageBlur(k=(2, 5)),  # 均值模糊
                iaa.MedianBlur(k=(3, 7)),  # 中值模糊
            ])),
            # sometimes(iaa.Pad(px=((0, 5), (0, 10), (0, 5), (0, 10)), pad_cval=255)),
            # sometimes(iaa.Crop(px=((0, 3), (0, 3), (0, 3), (0, 3)))),
            sometimes(iaa.PiecewiseAffine(
                scale=(0.01, 0.03), cval=255)),  # 扭曲图像局部区域
            sometimes(iaa.CLAHE(clip_limit=(3, 8))),
        ])
    ])
)


# img_aug = seq.augment_image(img)
if __name__ == "__main__":
    pass
    # import glob
    # import cv2
    # import os
    # from data_loader import resize_pad

    # folder = 'F:\\Datasets\\test_dataset'
    # img_paths = glob.glob('F:\\Datasets\\test_dataset\\*.jpg')
    # for img_path in img_paths:
    #     img_name = os.path.split(img_path)[1]
    #     img_save = os.path.join(
    #         'F:\\Datasets\\test_dataset\\imgaug_check', img_name)

    #     img = cv2.imread(img_path)
    #     img_aug = seq.augment_image(img)
    #     img_rp = resize_pad(img_aug, 160, 240)

    #     cv2.imwrite(img_save, img_rp)
    #     print(img_save, '--saved')
