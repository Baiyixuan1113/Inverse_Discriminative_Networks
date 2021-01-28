import copy
import os
import random

import numpy as np

num_people = 1050  # 数据人数
num_original = 50  # 正样本数量
num_forgeries = 50  # 负样本数量

dataset_path = '/home/data/Datasets/HWDSV_data'
save_path = '/home/DeepLeaning/project_IDN/data'
train_data_file = 'train_data.csv'
val_data_file = 'val_data.csv'
test_data_file = 'test_data.csv'


def combine(l, k):
    """生成从一个数组l中任取k个数的全部组合"""
    answers = []
    one = [0] * k

    def next_c(li=0, ni=0):
        if ni == k:
            answers.append(copy.copy(one))
            return
        for lj in range(li, len(l)):
            one[ni] = l[lj]
            next_c(lj + 1, ni + 1)

    next_c()
    return answers


def combine_2list(list1, list2):
    """生成从数组l1与数组l2中分别各取一个数的全部组合"""
    answers = []
    for i1 in list1:
        for i2 in list2:
            answers.append([i1, i2])
    return answers


def generate_df(num_people):
    train_size = int(num_people*0.9)
    val_size = int(num_people*0.08)
    test_size = num_people-train_size-val_size
    
    signature_list = list(range(1, num_people+1))
    lis_train = random.sample(signature_list, train_size)
    lis_val_test = list(set(signature_list).difference(set(lis_train)))
    lis_val = random.sample(lis_val_test, val_size)
    lis_test = list(set(lis_val_test).difference(set(lis_val)))
    
    org_path = os.path.join(dataset_path, 'full_org', 'original_')
    forg_path = os.path.join(dataset_path, 'full_forg', 'forgeries_')

    index_count_train = 1
    index_count_val = 1
    index_count_test = 1

    f_train = open(os.path.join(save_path, train_data_file), 'w')
    f_val = open(os.path.join(save_path, val_data_file), 'w')
    f_test = open(os.path.join(save_path, test_data_file), 'w')

    for i, sig in enumerate(signature_list):
        # print(i, sig)
        i = i+1
        org_org_lis = combine(list(range(1, num_original+1)), 2)
        random.shuffle(org_org_lis)
        org_forg_lis = combine_2list(list(range(1, num_original + 1)),
                                     list(range(1, num_forgeries + 1)))
        org_forg_lis = random.sample(org_forg_lis, len(org_org_lis))

        if i in train_indexs:
            # append train_data
            for item in org_org_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                org_1 = "%s%d_%d%s" % (org_path, int(sig), item[1], '.png')
                train_data_1 = '{},{},{}\n'.format(org_0, org_1, '1')
                f_train.write(train_data_1)
                index_count_train += 1

            for item in org_forg_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                forg_0 = "%s%d_%d%s" % (forg_path, int(sig), item[1], '.png')
                train_data_0 = '{},{},{}\n'.format(org_0, forg_0, '0')
                f_train.write(train_data_0)
                index_count_train += 1

        elif i in val_indexs:
            # append val_data
            for item in org_org_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                org_1 = "%s%d_%d%s" % (org_path, int(sig), item[1], '.png')
                val_data_1 = '{},{},{}\n'.format(org_0, org_1, '1')
                f_val.write(val_data_1)
                index_count_val += 1

            for item in org_forg_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                forg_0 = "%s%d_%d%s" % (forg_path, int(sig), item[1], '.png')
                val_data_0 = '{},{},{}\n'.format(org_0, forg_0, '0')
                f_val.write(val_data_0)
                index_count_val += 1

        elif i in test_indexs:
            # append test_data
            for item in org_org_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                org_1 = "%s%d_%d%s" % (org_path, int(sig), item[1], '.png')
                test_data_1 = '{},{},{}\n'.format(org_0, org_1, '1')
                f_test.write(test_data_1)
                index_count_test += 1

            for item in org_forg_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                forg_0 = "%s%d_%d%s" % (forg_path, int(sig), item[1], '.png')
                test_data_0 = '{},{},{}\n'.format(org_0, forg_0, '0')
                f_test.write(test_data_0)
                index_count_test += 1

    f_train.close()
    f_val.close()
    f_test.closer()

if __name__ == "__main__":
    generate_df(num_people)
