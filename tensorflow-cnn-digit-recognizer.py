import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import csv

# 1.load data

train_data = []
train_label = []
openfile = open('./train.csv','r')
reader = csv.reader(openfile)
for line in reader:
    train_data.append(line)
train_data =np.array(train_data)
train_label = train_data[1:,0]
train_data = train_data[1:,1:]
train_data = np.array(train_data).astype('float')
train_label = np.array(train_label).astype('float')

print(train_data.shape)
train_data = train_data.reshape(-1,28,28,1)
print(train_data.shape)
# print(train_label.shape)
test_data = []
openfile = open('./test.csv','r')
reader = csv.reader(openfile)
for line in reader:
    test_data.append(line)
test_data =np.array(test_data)
test_data = test_data[1:,:]
test_data = np.array(test_data).astype('float')
test_data = test_data.reshape(-1,28,28,1)
print(test_data.shape)


train_data,val_data,train_label,val_label = train_test_split(train_data,train_label,test_size=0.1,random_state=2)

# 2.define the model
model = tf.keras.Sequential([
    layers.Conv2D(input_shape=((28,28,1)),filters=32, kernel_size=(3,3),strides=(1,1), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.BatchNormalization(),
    layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])


# 优化器，别人的代码中使用了 RMSprop,在定义些操作，可以实现learning_rate的动态减小，开始学习率较大就可以比较快速的运算收敛，然后在准确率增长
# 或者不增长时再将学习率减小，这样就可以继续提高学习率。让学习在尽可能快的情况下提高准确率
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['acc'])
model.summary()


# 3.看到别人在kaggle上的分享，可以增加数据增强，也就是多创造一批训练数据，可以让模型拟合的更加完美了
# Data augmentation
# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(train_data)
history = model.fit_generator(datagen.flow(train_data,train_label,batch_size=128),epochs=10,validation_data=(val_data,val_label),
                              verbose=2)
# history = model.fit(train_data,train_label,epochs=5,batch_size=128,validation_split=0.1)

pred = model.predict(test_data)
pred= np.argmax(pred,axis=1)

print(history)
print(pred)
print(pred.shape)
plt.plot(history.epoch,history.history.get('loss'))
plt.show()
plt.plot(history.epoch,history.history.get('acc'))
plt.show()
label =['ImageId','Label']
predd = open("./sub.csv",'w',newline='')
writer = csv.writer(predd)
writer.writerow(label)
length = len(pred)
for i in range(length):
    predict=[]
    predict.append(i+1)
    predict.append(pred[i])
    writer.writerow(predict)


