import os
import sys
import time

import tensorflow as tf

root_path = os.getcwd()
sys.path.append(os.path.join(root_path, 'code'))
from config import *
from data_loader import get_image_array
from IDN import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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


def predict_by_pb(img_01_path,
                  img_02_path,
                  input_height,
                  input_width,
                  pb_model_path,
                  loss_weight=[0.3, 0.4, 0.3]):
    """
    单次预测
    :param img_01_path: 测试图像01的路径
    :param img_02_path: 测试图像02的路径
    :param input_height: 模型输入尺寸高
    :param input_width: 模型输入尺寸宽
    :param pb_model_path: pb模型路径
    :param loss_weight: 计算输出结果的权重比例
    """
    op_time_model = time.time()
    with tf.gfile.FastGFile(pb_model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,
                            input_map=None,
                            return_elements=None,
                            name="",
                            op_dict=None,
                            producer_op_list=None)
    ed_time_model = time.time()
    cost_time_model = ed_time_model-op_time_model
    print('----Graph loaded---- (TIME:{:.4}s)'.format(cost_time_model))

    img_01 = processong_img(img_path=img_01_path,
                            input_height=input_height,
                            input_width=input_width)
    img_02 = processong_img(img_path=img_02_path,
                            input_height=input_height,
                            input_width=input_width)

    op_time = time.time()
    with tf.Session() as sess:
        img_input_01 = sess.graph.get_tensor_by_name('input_1:0')
        img_input_02 = sess.graph.get_tensor_by_name('input_2:0')
        output_01 = sess.graph.get_tensor_by_name('output_1:0')
        output_02 = sess.graph.get_tensor_by_name('output_2:0')
        output_03 = sess.graph.get_tensor_by_name('output_3:0')

        result = sess.run([output_01, output_02, output_03],
                          feed_dict={img_input_01: img_01,
                                     img_input_02: img_02})
        # print(result)

    ed_time = time.time()
    cost_time = ed_time-op_time

    k0, k1, k2 = loss_weight
    s0 = result[0].tolist()[0][0]
    s1 = result[1].tolist()[0][0]
    s2 = result[2].tolist()[0][0]
    score = k0*s0+k1*s1+k2*s2
    return score, cost_time


def predict_by_pb_volume(input_height,
                         input_width,
                         pb_model_path,
                         test_data_path,
                         test_result_path=None,
                         loss_weight=[0.3, 0.4, 0.3]):
    """
    批量预测
    :param input_height: 模型输入尺寸高
    :param input_width: 模型输入尺寸宽
    :param pb_model_path: pb模型路径
    :param test_data_path: 测试数据文件路径
    :param test_result_path: 测试结果保存路径，默认为None，不保存
    :param loss_weight: 计算输出结果的权重比例
    """

    op_time_model = time.time()
    with tf.gfile.FastGFile(pb_model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,
                            input_map=None,
                            return_elements=None,
                            op_dict=None,
                            producer_op_list=None)
    ed_time_model = time.time()
    cost_time_model = ed_time_model-op_time_model
    print('-------Graph loaded------- (TIME:{:.4}s)'.format(cost_time_model))

    data_list = get_list_for_data(test_data_path)
    data_txt = 'img_01, img_02, label_true, lable_pred\n'
    for img_01_path, img_02_path, label_t in data_list:
        # 读入图像并进行预处理
        img_01 = processong_img(img_path=img_01_path,
                                input_height=input_height,
                                input_width=input_width)
        img_02 = processong_img(img_path=img_02_path,
                                input_height=input_height,
                                input_width=input_width)
        # 预测
        op_time = time.time()
        with tf.Session() as sess:
            img_input_01 = sess.graph.get_tensor_by_name('input_1:0')
            img_input_02 = sess.graph.get_tensor_by_name('input_2:0')
            output_01 = sess.graph.get_tensor_by_name('output_1:0')
            output_02 = sess.graph.get_tensor_by_name('output_2:0')
            output_03 = sess.graph.get_tensor_by_name('output_3:0')
            result = sess.run([output_01, output_02, output_03],
                              feed_dict={img_input_01: img_01,
                                         img_input_02: img_02})
        ed_time = time.time()
        cost_time = ed_time-op_time

        k0, k1, k2 = loss_weight
        s0 = result[0].tolist()[0][0]
        s1 = result[1].tolist()[0][0]
        s2 = result[2].tolist()[0][0]
        label_p = k0*s0+k1*s1+k2*s2
        line = '{}, {}, {}, {}\n'.format(
            img_01_path, img_02_path, label_t, label_p)
        data_txt += line
        print('==== label_true:{}; label_pred:{:4f}; cost_time:{:.4}s'.format(
            label_t, label_p, cost_time))

    if test_result_path:
        with open(test_result_path, 'w') as f:
            f.write(data_txt)
            print('The test result has been saved to ', test_result_path)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    input_width = INPUT_WIDTH  # 输入尺寸的宽
    input_height = INPUT_HEIGHT  # 输入尺寸的高
    input_channel = INPUT_CHANNEL  # 输入尺寸的通道数
    loss_weight = [0.3, 0.4, 0.3]  # 损失函数权重比例
    # threshold = 0.5  # 判定准确率阈值

    test_img_01 = 'F:\\Deep_learning\\IDN_keras\\asdzz\\org_01_01.jpg'
    test_img_02 = 'F:\\Deep_learning\\IDN_keras\\asdzz\\org_01_02.jpg'
    test_img_03 = 'F:\\Deep_learning\\IDN_keras\\asdzz\\forg_01_01.jpg'

    # ------------------------------     .H5模型路径     ------------------------------
    pb_path = 'F:\\Deep_learning\\IDN_keras\\pb_models\\model_IDN.pb'
    # --------------------------------------------------------------------------------

    # ------------------------------  test_data.csv路径  ------------------------------
    # test_data_path = '/home/data/ETOP/project_IDN/data_check_0118/test_data.csv'
    # test_data_path = '/home/data/ETOP/project_IDN/test_data_old/test_old_photo.csv'
    # test_data_path = '/home/data/ETOP/project_IDN/test_data_old/test_old_black.csv'
    # --------------------------------------------------------------------------------

    # ------------------------------ test_result.csv路径 ------------------------------
    # test_result_path = '/home/data/ETOP/project_IDN/test_result/test_result_01.csv'
    # test_result_path = '/home/data/ETOP/project_IDN/test_result/test_result_02.csv'
    # test_result_path = '/home/data/ETOP/project_IDN/test_result/test_result_03.csv'
    # --------------------------------------------------------------------------------

    score, cost_time = predict_by_pb(img_01_path=test_img_01,
                                     img_02_path=test_img_03,
                                     input_height=input_height,
                                     input_width=input_width,
                                     pb_model_path=pb_path,
                                     loss_weight=loss_weight)
    print('predict_result:{:4f}; cost_time:{:.4}s'.format(score, cost_time))
