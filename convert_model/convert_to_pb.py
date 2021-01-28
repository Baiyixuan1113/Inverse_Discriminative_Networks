import os
import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import models
from keras.backend import *
from keras.layers import *
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.optimizers import SGD, Adam
from tensorflow.python.framework import graph_io, graph_util
from tensorflow.python.tools import import_pb_to_tensorboard

root_path = os.getcwd()
sys.path.append(os.path.join(root_path, 'code'))
from config import *
from IDN import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_h5_model(model_path,
                  input_height,
                  input_width,
                  input_channel):
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

    model.load_weights(model_path)
    print('**** loading complete.')

    return model


def h5_to_pb(h5_model,
             output_dir,
             model_name,
             out_prefix="output_",
             log_tensorboard=True):
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []

    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))

    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(
        sess,
        init_graph,
        out_nodes)
    graph_io.write_graph(main_graph,
                         output_dir,
                         name=model_name,
                         as_text=False)
    if log_tensorboard:
        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir, model_name), output_dir)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    input_width = INPUT_WIDTH  # 输入尺寸的宽
    input_height = INPUT_HEIGHT  # 输入尺寸的高
    input_channel = INPUT_CHANNEL  # 输入尺寸的通道数

    # .H5模型路径
    model_path = 'F:\\Deep_learning\\IDN_keras\\logs\\model_905.h5'
    model_save_path = 'F:\\Deep_learning\\IDN_keras\\pb_models'
    model_save_name = 'model_IDN.pb'

    # 加载模型
    model = load_h5_model(model_path=model_path,
                          input_height=input_height,
                          input_width=input_width,
                          input_channel=input_channel)

    h5_to_pb(model, output_dir=model_save_path, model_name=model_save_name)
    print('model saved.')
