import os
import time

import numpy as np
import tensorflow as tf
from keras import models
from keras.backend import *
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.optimizers import SGD, Adam

from config import *
from data_loader import data_generate, get_image_array
from IDN import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
            data_list.append((file_path_01, file_path_02, label))
            # print(file_path_01, file_path_02, label)
    return data_list


def processong_img(img_path, input_height, input_width):
    img = get_image_array(img_path,
                          height=input_height,
                          width=input_width,
                          is_training=False,
                          do_augment=False)

    img = img.reshape((1, input_height, input_width, 3))
    return img


def load_h5_model(model_path,
                  input_height,
                  input_width,
                  input_channel,
                  loss_weight):
    print('**** loading model...')

    input_a_shape = (input_height, input_width, input_channel)
    input_b_shape = (input_height, input_width, input_channel)

    input_a_tensor = Input(shape=input_a_shape)
    input_b_tensor = Input(shape=input_b_shape)

    outputs_a = IDN_discriminative_stream(input_a_tensor)
    outputs_b = IDN_discriminative_stream(input_b_tensor)

    inv_stm_a, dis_stm_a = outputs_a
    inv_stm_b, dis_stm_b = outputs_b

    v1 = feature_merge([inv_stm_a, dis_stm_b])
    v2 = feature_merge([dis_stm_a, dis_stm_b])
    v3 = feature_merge([inv_stm_b, dis_stm_a])
    result = [v1, v2, v3]

    # result = Lambda(get_output, arguments={'loss_weight': loss_weight})([v1, v2, v3])

    model = models.Model(inputs=[input_a_tensor, input_b_tensor],
                         outputs=result, name='IDN')

    model.compile(optimizer=Adam(lr=0.001),
                  loss=binary_crossentropy,
                  loss_weights=loss_weight,
                  metrics=[binary_accuracy])

    model.load_weights(model_path)
    print('**** loading complete.')

    return model


def predict_by_h5(model, input_imgs, loss_weight):
    k1, k2, k3 = loss_weight
    op_time = time.time()
    v1, v2, v3 = model.predict(input_imgs)
    ed_time = time.time()

    result = k1*v1+k2*v2+k3*v3
    result = result.tolist()[0][0]
    cost_time = ed_time-op_time
    # print('==== result:{:.4}, time:{:.4}s'.format(result, cost_time))
    return result, cost_time


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    input_width = INPUT_WIDTH  # 输入尺寸的宽
    input_height = INPUT_HEIGHT  # 输入尺寸的高
    input_channel = INPUT_CHANNEL  # 输入尺寸的通道数
    loss_weight = [0.3, 0.4, 0.3]  # 损失函数权重比例
    # threshold = 0.5  # 判定准确率阈值

    # ------------------------------     .H5模型路径     ------------------------------
    model_path = ''
    # --------------------------------------------------------------------------------

    # ------------------------------  test_data.csv路径  ------------------------------
    test_data_path = ''
    # --------------------------------------------------------------------------------

    # ------------------------------ test_result.csv路径 ------------------------------
    test_result_path = ''
    # --------------------------------------------------------------------------------

    data_list = get_list_for_data(test_data_path)
    label_true = []
    label_pred = []

    f_test_re = open(test_result_path, 'w')
    f_test_re.write('img_01, img_02, label_true, lable_pred\n')

    # 加载模型
    model = load_h5_model(model_path=model_path,
                          input_height=input_height,
                          input_width=input_width,
                          input_channel=input_channel,
                          loss_weight=loss_weight)

    for img_01_path, img_02_path, label_t in data_list:
        # 读入图像并进行预处理
        img_01 = processong_img(img_path=img_01_path,
                                input_height=input_height,
                                input_width=input_width)
        img_02 = processong_img(img_path=img_02_path,
                                input_height=input_height,
                                input_width=input_width)
        # 预测
        label_p, cost_time = predict_by_h5(model=model,
                                           input_imgs=[img_01, img_02],
                                           loss_weight=loss_weight)

        print('==== label_true:{:.2f}; label_pred:{:4f}; cost_time:{:.4}s'.format(
            float(label_t), float(label_p), cost_time))

        label_true.append(label_t)
        label_pred.append(label_p)
        data = '{}, {}, {}, {}\n'.format(
            img_01_path, img_02_path, label_t, label_p)
        f_test_re.write(data)

    f_test_re.close()
