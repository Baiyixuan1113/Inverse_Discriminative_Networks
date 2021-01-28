"""
计算准确率
"""

# ------------------------------   测试结果文件路径    ------------------------------
# file_path = '/home/data/ETOP/project_IDN/test_result/test_result_01.csv'
# file_path = '/home/data/ETOP/project_IDN/test_result/test_result_02.csv'
file_path = '/home/data/ETOP/project_IDN/test_result/test_result_03.csv'
# --------------------------------------------------------------------------------

# ------------------------------   预测错误样本对记录   ------------------------------
# test_wrong_path = '/home/data/ETOP/project_IDN/test_result/test_wrong_01.csv'
# test_wrong_path = '/home/data/ETOP/project_IDN/test_result/test_wrong_02.csv'
test_wrong_path = '/home/data/ETOP/project_IDN/test_result/test_wrong_03.csv'
# --------------------------------------------------------------------------------

threshold = 0.5
org_acc_lis = []
forg_acc_lis = []
with open(file_path) as f:
    lines = f.readlines()

with open(test_wrong_path, 'w') as f_c:
    f_c.write(lines[0])

    for line in lines[1:]:
        label_true = int(line.strip('\n').split(',')[-2])
        label_pred = float(line.strip('\n').split(',')[-1])

        if label_pred >= threshold:
            label_pred = 1
        elif label_pred < threshold:
            label_pred = 0

        if label_true == 1:
            if label_pred == label_true:
                org_acc_lis.append(1)
            elif label_pred != label_true:
                org_acc_lis.append(0)
                f_c.write(line)
                # print(line)
        elif label_true == 0:
            if label_pred == label_true:
                forg_acc_lis.append(1)
            elif label_pred != label_true:
                forg_acc_lis.append(0)
                f_c.write(line)
                # print(line)


print('数据量：', len(org_acc_lis) + len(forg_acc_lis))
print('正样本对数量：', len(org_acc_lis))
print('负样本对数量：', len(forg_acc_lis))

print('总准确率:{:.4f}'.format((sum(org_acc_lis)+sum(forg_acc_lis))/(len(org_acc_lis)+len(forg_acc_lis))))
print('正样本准确率:{:.4f}'.format(sum(org_acc_lis)/len(org_acc_lis)))
print('负样本准确率:{:.4f}'.format(sum(forg_acc_lis)/len(forg_acc_lis)))
