# kaggle上面的入门题 手写数字识别
# 每张图是28*28 = 784个像素点，所以一张图片的输入是（784，1）

import numpy as np
import csv

def loadtrainfile(file):
    train_data = []
    train_label = []
    openfile = open(file,'r')
    read = csv.reader(openfile)
    for line in read:
        train_data.append(line)
        # print(line)
    train_data=np.array(train_data)
    train_label = train_data[:,0]
    train_data = train_data[1:,1:]
    train_label=train_label[1:]
    return (np.array(train_data),np.array(train_label))


def loadtestfile(file):
    test_data = []
    test_label = []
    openfile = open(file,'r')
    read = csv.reader(openfile)
    for line in read:
        test_data.append(line)
        # print(line)
    test_data=np.array(test_data)
    test_data = test_data[1:,:]

    return np.array(test_data)

def toInt_Normalize(array):
    m=array.shape[0]
    n=array.shape[1]
    print(m,n)
    new_array = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            # array[i][j]=int(array[i][j])   # 这里不能进行改动，怪不得人家的代码要新建一个二维数组
            if int(array[i][j])!=0:
                new_array[i][j]=1
            else:
                new_array[i][j]=0
    return new_array
    # for line in array:
    #     for item in line:
    #         item = int(item)
    #         if item!=0:
    #             item =1
    # return array


def labelToInt_Normalize(array):
    # 搞清楚这里是要把测试机数据拿来变成整数和归一化
    m = array.shape[0]
    n = array.shape[1]
    print(m, n)
    new_array = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            # array[i][j]=int(array[i][j])   # 这里不能进行改动，怪不得人家的代码要新建一个二维数组
            if int(array[i][j]) != 0:
                new_array[i][j] = 1
            else:
                new_array[i][j] = 0
    return new_array


def knn_neighbor(train_data,inx,train_label,k):
    #inx是test数据，train_data是比较的训练数据,train_label是训练集标签数据
    num_test = inx.shape[0]
    print("num_test:"+str(num_test))
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        distances = np.sum(np.abs(train_data-inx[i,:]),axis=1)
        print(len(distances))
        kmin = []
        for j in range(k):
            min_index = np.argmin(distances)
            kmin.append(train_label[min_index])
            max_index = np.argmax(distances)
            distances[min_index]= distances[max_index]
        b =np.argmax(np.bincount(kmin))
        y_pred[i]=b
        if i % 10 == 0:
            print("第"+str(i)+"步")
    return y_pred

train_data ,train_label = loadtrainfile('./train.csv')
train_data = toInt_Normalize(train_data)
print(train_data[1])
print("-----------")


test_data = loadtestfile('./test.csv')
test_data = labelToInt_Normalize(test_data)
print(test_data[1])
y_pred = knn_neighbor(train_data,test_data,train_label,10)
print(y_pred)
label = ['ImageId','Label']
pred = open("./sub.csv",'w',newline='')
writer = csv.writer(pred)
writer.writerow(label)
length =len(y_pred)
for i in range(length):
    predict =[]
    predict.append(i)
    predict.append(y_pred[i])
    writer.writerow(predict)


