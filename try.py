import numpy as np
import csv
a = np.array([['1','2','3'],
              ['3','3','3'],
              ['2','3','4']])

# 寻找二维数组中每行最大的值
print(a.shape)
b = np.argmax(a,axis=1)
b =a[range(a.shape[0]),b]
print(b)


print(type(a[1][1]))
a[1][1]=int(a[1][1])
print(type(a[1][1]))
print(a)

label = ['ImageId','Label']
y_pred = [1,2,3,4,5,56,6,7,6]
pred = open("./sub.csv",'w',newline='')
writer = csv.writer(pred)
writer.writerow(label)
writer.writerow(y_pred)
