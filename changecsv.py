# 修改一下sub.csv的输出 输出序号从1开始，然后label是整数
import csv
import numpy as np
train_data = []
openfile = open('./sub.csv', 'r')
read = csv.reader(openfile)
for line in read:
    train_data.append(line)
    # print(line)
train_data = np.array(train_data)