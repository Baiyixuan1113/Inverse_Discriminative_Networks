import copy
import os
import random
import sys

import imageio
import numpy as np
import pandas as pd

num_people = 1050  # 数据人数
num_original = 50  # 正样本数量
num_forgeries = 50  # 负样本数量

dataset_path = '/home/data/ETOP_HWDS_data'
save_path = '/home/data/project_IDN/data'
train_data_file = 'train_data.csv'
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


def generate_df(train_size):
    signature_list = list(range(1, num_people))
    train_indexs = np.arange(1, len(signature_list)+1, 1)
    np.random.shuffle(train_indexs)
    train_indexs = train_indexs[:train_size + 1]
    a = [1, 22, 43, 64, 85,
         106, 127, 148, 169, 190,
         211, 232, 253, 274, 295,
         316, 337, 358, 379, 400,
         421, 442, 463, 484, 505,
         526, 547, 568, 589, 610,
         631, 652, 673, 694, 715,
         736, 757, 778, 799, 820,
         841, 862, 883, 904, 925,
         946, 967, 988, 1009, 1030]
    b = random.sample(a, 45)
    b.sort()
    train_indexs = []
    for xx in b:
        for yy in range(21):
            train_indexs.append(xx+yy)

    org_path = os.path.join(dataset_path, 'full_org', 'original_')
    forg_path = os.path.join(dataset_path, 'full_forg', 'forgeries_')
    train_df = pd.DataFrame(columns=['image_1', 'image_2', 'label'])
    val_df = pd.DataFrame(columns=['image_1', 'image_2', 'label'])
    index_count_train = 1
    index_count_val = 1

    for i, sig in enumerate(signature_list):
        org_org_lis = combine(list(range(1, num_original+1)), 2)
        random.shuffle(org_org_lis)
        org_forg_lis = combine_2list(list(range(1, num_original + 1)),
                                     list(range(1, num_forgeries + 1)))
        org_forg_lis = random.sample(org_forg_lis, len(org_org_lis))

        if i in train_indexs:
            print('train_indexs:', sig)
            for item in org_org_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                org_1 = "%s%d_%d%s" % (org_path, int(sig), item[1], '.png')
                train_df.loc[index_count_train] = [org_0, org_1, '1']
                index_count_train += 1

            for item in org_forg_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                forg_0 = "%s%d_%d%s" % (forg_path, int(sig), item[1], '.png')
                train_df.loc[index_count_train] = [org_0, forg_0, '0']
                index_count_train += 1
        else:
            print('val_indexs:', sig)
            for item in org_org_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                org_1 = "%s%d_%d%s" % (org_path, int(sig), item[1], '.png')
                val_df.loc[index_count_val] = [org_0, org_1, '1']
                index_count_val += 1

            for item in org_forg_lis:
                org_0 = "%s%d_%d%s" % (org_path, int(sig), item[0], '.png')
                forg_0 = "%s%d_%d%s" % (forg_path, int(sig), item[1], '.png')
                val_df.loc[index_count_val] = [org_0, forg_0, '0']
                index_count_val += 1

    print(train_df.shape)
    print(val_df.shape)
    train_df.to_csv(os.path.join(save_path, train_data_file), index=False)
    val_df.to_csv(os.path.join(save_path, test_data_file), index=False)


if __name__ == "__main__":
    generate_df(train_size=420)
