import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
import numpy as np
import time


batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('x[0]_train shape:', x_train[0].shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])


model = Sequential()
model.add(Conv2D(8, kernel_size=(5, 5),strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

model.add(Conv2D(16, (5, 5), strides=(1, 1) ,activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(3, 3)))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.load_weights("model_weights_5_convnetjs.h5")

def predict_plate(list_angka, mnist=True):
    if mnist:
        prediksi = ""
        for i in range(len(list_angka)):
            img = list_angka[i]
            ret, img= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            img = img.reshape(28,28,1)
            img = img.astype('float32')
            img /= 255

            prediction = model.predict(np.array([img]))
            hasil = prediction.argmax(axis=1)
            prediksi = prediksi + str(hasil[0])

        return prediksi

    else:
        prediksi = ""
        for i in range(len(list_angka)):
            img = list_angka[i]

            templates = []
            angka0 = cv2.imread('angka/0.jpg', 0)
            angka0 = cv2.resize(angka0,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka0)
            angka1 = cv2.imread('angka/1.jpg', 0)
            angka1 = cv2.resize(angka1,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka1)
            angka2 = cv2.imread('angka/2.jpg', 0)
            angka2 = cv2.resize(angka2,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka2)
            angka3 = cv2.imread('angka/3.jpg', 0)
            angka3 = cv2.resize(angka3,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka3)
            angka4 = cv2.imread('angka/4.jpg', 0)
            angka4 = cv2.resize(angka4,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka4)
            angka5 = cv2.imread('angka/5.jpg', 0)
            angka5 = cv2.resize(angka5,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka5)
            angka6 = cv2.imread('angka/6.jpg', 0)
            angka6 = cv2.resize(angka6,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka6)
            angka7 = cv2.imread('angka/7.jpg', 0)
            angka7 = cv2.resize(angka7,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka7)
            angka8 = cv2.imread('angka/8.jpg', 0)
            angka8 = cv2.resize(angka8,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka8)
            angka9 = cv2.imread('angka/9.jpg', 0)
            angka9 = cv2.resize(angka9,(28,28), interpolation = cv2.INTER_AREA)
            templates.append(angka9)

            angka = -1
            angka_val = float("inf") 

            ii = 0
            for template in templates:
                method = eval('cv2.TM_SQDIFF')
                img2 = img.copy()
                # Apply template Matching
                res = cv2.matchTemplate(img2, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if min_val < angka_val:
                    angka_val = max_val
                    angka = ii
                print str(ii)+": "+str(max_val)
                ii+=1

            print angka_val
            pred =  "["+str(angka)+"]"
            prediksi += pred

        return prediksi

