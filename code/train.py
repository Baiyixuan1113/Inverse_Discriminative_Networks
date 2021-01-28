import os
import time
from contextlib import redirect_stdout

import keras
import tensorflow as tf
from keras import models
from keras.backend import *
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import *
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.optimizers import SGD, Adam
# from keras.utils import plot_model

from config import *
from data_loader import data_generate
from IDN import *

# PROJECT_NAME
project_name = "IDN_net"

# ROOT_PATH
project_path = '/home/data/project_IDN'

# DATA_FILE
data_file = '/home/data/project_IDN/data'

# MODEL_SAVE
model_save_path = os.path.join(project_path, 'logs', project_name)
csvloger_path = '/home/data/project_IDN/trainval_note/{}.csv'.format(project_name)

# 是否加载预训练模型
need_load_weight = False  # True or False
load_weight_path = ''

if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    data_file = data_file  # 划分完成的数据集文件路径
    model_save = model_save_path  # 保存模型的路径
    input_width = INPUT_WIDTH  # 输入尺寸的宽
    input_height = INPUT_HEIGHT  # 输入尺寸的高
    input_channel = INPUT_CHANNEL  # 输入尺寸的通道数
    epochs = EPOCHS  # 轮次
    batch_size = BATCH_SIZE  # 批次大小
    steps_per_epoch = STEPS_PER_EPOCH
    validation_steps = VALIDATION_STEPS
    loss_weight = [0.3, 0.4, 0.3]  # 损失函数权重比例
    # threshold = 0.5  # 判定准确率阈值

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

    # 加载预训练模型
    if need_load_weight:
        print('**********loadeing model from:', load_weight_path)
        print('...')
        model.load_weights(load_weight_path)
        print('...', '...')
        print('**********loading complete.')
    else:
        print('**********There is no pre-training model that needs to be loaded.')

    # ------------------------------------------------------------
    # 保存模型结构到txt文件
    # with open('/home/data/project_IDN/IDN.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         model.summary()

    # ------------------------------------------------------------
    # 保存模型结构到png文件
    # plot_model(model=model,
    #            to_file='/home/data/project_IDN/IDN.png')

    train_data_gen = data_generate(data_list_path=data_file,
                                   batch_size=batch_size,
                                   height=input_height,
                                   width=input_width,
                                   is_training=True,
                                   do_augment=True)

    val_data_gen = data_generate(data_list_path=data_file,
                                 batch_size=batch_size,
                                 height=input_height,
                                 width=input_width,
                                 is_training=False,
                                 do_augment=False)

    if not os.path.exists(model_save):
        os.makedirs(model_save)

    checkpointer = ModelCheckpoint(
        os.path.join(model_save, 'model_{epoch:04d}.h5'),
        verbose=1,
        save_weights_only=False,
        period=1
    )

    csvloger_a = CSVLogger(csvloger_path, append=True)

    model.fit_generator(generator=train_data_gen,
                        validation_data=val_data_gen,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        use_multiprocessing=False,
                        callbacks=[checkpointer, csvloger_a])
