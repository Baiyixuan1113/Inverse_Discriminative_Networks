from contextlib import redirect_stdout

import keras
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from keras.layers import *
from keras.utils import plot_model

from config import IMAGE_ORDERING

if IMAGE_ORDERING == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1


def img_fanzhuan(tensor):
    """像素值反转"""
    result = 255 - tensor
    return result


def attention_module(inverse_stream, discrimnative_stream, filters):
    """attention模块"""
    m = UpSampling2D(size=2, interpolation="nearest")(inverse_stream)
    m = Conv2D(filters, (3, 3), strides=1, padding="same")(m)
    m = BatchNormalization(axis=bn_axis, scale=False)(m)
    g = Activation("sigmoid")(m)
    h = discrimnative_stream
    n = Multiply()([h, g])  # n=h·g
    n = Add()([n, h])  # n=h·g+h
    f = GlobalAveragePooling2D()(n)  # f=GAP(h·g+f)
    f = Dense(filters, activation="sigmoid")(f)  # f=GAP(h·g+f) --> FC(sigmoid)
    n = Multiply()([n, f])  # n=(h·g+h)×f
    # print('======[(h·g+h)×f]-shape-:', n)
    return n


def IDN_inverse_stream(inp_tensor):

    inverse_input = inp_tensor
    # inverse_input = img_fanzhuan(inp_tensor)
    x = Lambda(img_fanzhuan)(inverse_input)

    # f1-stage
    x = Conv2D(32, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    f1 = MaxPool2D((2, 2), strides=(2, 2))(x)

    # f2-stage
    x = Conv2D(64, (3, 3), strides=1, padding="same")(f1)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    f2 = MaxPool2D((2, 2), strides=(2, 2))(x)

    # f3-stage
    x = Conv2D(96, (3, 3), strides=1, padding="same")(f2)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(96, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    f3 = MaxPool2D((2, 2), strides=(2, 2))(x)

    # f4-stage
    x = Conv2D(128, (3, 3), strides=1, padding="same")(f3)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    f4 = MaxPool2D((2, 2), strides=(2, 2))(x)

    f = [f1, f2, f3, f4]
    return f
    # model = models.Model(inputs=inverse_input, outputs=f4, name="IDN_inverse_stream")
    # return model


def IDN_discriminative_stream(inp_tensor):

    f = IDN_inverse_stream(inp_tensor)
    f1, f2, f3, f4 = f

    discriminative_input = inp_tensor

    x = Conv2D(32, (3, 3), strides=1, padding="same")(discriminative_input)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    # attention_module_01
    x = attention_module(filters=32,
                         inverse_stream=f1,
                         discrimnative_stream=x)

    x = Conv2D(32, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    # attention_module_02
    x = attention_module(filters=64, 
                         inverse_stream=f2,
                         discrimnative_stream=x)

    x = Conv2D(64, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(96, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    # attention_module_03
    x = attention_module(filters=96, 
                         inverse_stream=f3,
                         discrimnative_stream=x)

    x = Conv2D(96, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    # attention_module_04
    x = attention_module(filters=128, 
                         inverse_stream=f4,
                         discrimnative_stream=x)

    x = Conv2D(128, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    outputs = [x, f4]

    return outputs


def feature_merge(features):
    ft_01, ft_02 = features
    # ft_merge = K.concatenate([ft_01, ft_02], axis=-1)
    ft_merge = Concatenate(axis=-1)([ft_01, ft_02])

    x = Conv2D(256, (3, 3), strides=1, padding="same")(ft_merge)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), strides=1, padding="same")(ft_merge)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)

    x = Dense(256, activation="relu")(x)

    x = Dense(1, activation="sigmoid")(x)

    return x


def get_output(pred_values, loss_weight=[0.3, 0.4, 0.3]):
    """sigmoid + binary_crossentropy"""
    v1, v2, v3 = pred_values
    k1, k2, k3 = loss_weight

    output = K.sum([k1 * v1, k2 * v2, k3 * v3])
    # print('output:', output)
    return output


if __name__ == "__main__":
    pass
    # input_a_shape = (160, 240, 3)
    # input_b_shape = (160, 240, 3)
    # input_a_tensor = Input(shape=input_a_shape)
    # input_b_tensor = Input(shape=input_b_shape)
    # outputs_a = IDN_discriminative_stream(input_a_tensor)
    # outputs_b = IDN_discriminative_stream(input_b_tensor)

    # inv_stm_a, dis_stm_a = outputs_a
    # inv_stm_b, dis_stm_b = outputs_b

    # v1 = feature_merge([inv_stm_a, dis_stm_b])
    # v2 = feature_merge([dis_stm_a, dis_stm_b])
    # v3 = feature_merge([inv_stm_b, dis_stm_a])

    # # result = Lambda(get_output)([v1, v2, v3])
    # result = Lambda(get_output, arguments={
    #                 'loss_weight': [0.3, 0.4, 0.3]})([v1, v2, v3])

    # model = models.Model(inputs=[input_a_tensor, input_b_tensor],
    #                      outputs=result, name='IDN')

    # # ------------------------------------------------------------
    # # 保存模型结构到txt文件
    # with open('F:\\code-practising\\project_IDN\\code\\IDN.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         model.summary()

    # # ------------------------------------------------------------
    # # 保存模型结构到png文件
    # plot_model(model=model,
    #            to_file='F:\\code-practising\\project_IDN\\code\\IDN.png')
