import copy
import os
import random

import numpy as np

num_people = 1050  # 数据人数
num_original = 50  # 正样本数量
num_forgeries = 50  # 负样本数量

dataset_path = '/home/data/Datasets/HWDSV_data/HWDS_20201225'
save_path = '/home/data/ETOP/project_IDN/data'
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


def generate_df(train_size):
    signature_list = list(range(1, num_people+1))
    train_indexs = np.arange(1, len(signature_list)+1, 1)
    np.random.shuffle(train_indexs)
    train_indexs = train_indexs[:train_size + 1]

    #########################################
    # 按组别划分数据集
    # a = [1, 22, 43, 64, 85,
    #      106, 127, 148, 169, 190,
    #      211, 232, 253, 274, 295,
    #      316, 337, 358, 379, 400,
    #      421, 442, 463, 484, 505,
    #      526, 547, 568, 589, 610,
    #      631, 652, 673, 694, 715,
    #      736, 757, 778, 799, 820,
    #      841, 862, 883, 904, 925,
    #      946, 967, 988, 1009, 1030]
    # b = random.sample(a, 49)
    # c = list(set(a).difference(set(b)))
    # d = random.sample(b, 45)
    # e = list(set(b).difference(set(d)))

    # train_nums = d
    # val_nums = e
    # test_nums = c
    # train_nums.sort()
    # val_nums.sort()
    # test_nums.sort()
    #########################################
    # 1050组按组别划分数据集
    lis_a = list(range(50))
    lis_aa = [a*21+1 for a in lis_a]
    lis_b = list(range(13, 35))
    lis_bb = [b*21+1 for b in lis_b]

    lis_test = random.sample(lis_bb, 1)
    # 求差集，在lis_aa中但不在lis_test中
    lis_trainval = list(set(lis_aa).difference(set(lis_test)))
    lis_train = random.sample(lis_trainval, 45)
    lis_val = list(set(lis_trainval).difference(set(lis_train)))
    lis_train.sort()
    lis_val.sort()
    lis_test.sort()
    #########################################

    train_indexs = []
    val_indexs = []
    test_indexs = []
    for xx in lis_train:
        for yy in range(21):
            train_indexs.append(xx+yy)
    for xx in lis_val:
        for yy in range(21):
            val_indexs.append(xx+yy)
    for xx in lis_test:
        for yy in range(21):
            test_indexs.append(xx+yy)
    print('len_train:', len(train_indexs))
    print('len_val:', len(val_indexs))
    print('len_test:', len(test_indexs))
    #########################################

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


if __name__ == "__main__":
    generate_df(train_size=945)
