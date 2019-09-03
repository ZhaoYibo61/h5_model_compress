import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(10)

model_save_name = "cifar10CNN.h5"

(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()
print("size train data =", len(x_train_image))
print("size test data =", len(x_test_image))

print(x_train_image.shape)
print(x_train_image[0])
print(y_train_label.shape)

label_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer",
            5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

x_train_normal = x_train_image.astype("float32") / 255.0
x_test_normal = x_test_image.astype("float32") / 255.0

y_train_onehot = np_utils.to_categorical(y_train_label)
y_test_onehot = np_utils.to_categorical(y_test_label)

model = Sequential()

model.add(Conv2D(filters=32,
                kernel_size = (3, 3),
                padding = 'same',
                input_shape = (32, 32, 3),
                activation = 'relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,
                kernel_size = (3, 3),
                padding = 'same',
                activation = 'relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(units = 1024,
                activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 10,
                activation = 'softmax'))

print(model.summary())


model.compile(loss = "categorical_crossentropy",
            optimizer = "adam", metrics = ["accuracy"])

try:
    model.load_weights(model_save_name)
except:
    print("load model failed!")

history = model.fit(x = x_train_normal,
                y = y_train_onehot,
                validation_split = 0.2,
                epochs = 3,
                batch_size = 128,
                verbose = 2)

model.save(model_save_name)

def show_train_history(train_history, train, val):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[val])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epochs")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def plot_image_label_prediction(images, labels, prediction, idx = 0, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap="binary")
        title = str(i) + "." + label_dict[labels[idx][0]]
        if len(prediction) > 0:
            title += " => " + label_dict[prediction[idx]]
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

def show_prediction_prob(y_label, prediction, x_image, prediction_prob, i):
    print("label: ", label_dict[y_label[i][0]], " predict: ", label_dict[prediction[i]])
    plt.figure(figsize = (2, 2))
    plt.imshow(np.reshape(x_image[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print(label_dict[j], " predict probability: %1.9f" % (prediction_prob[i][j]))

show_train_history(history, "acc", "val_acc")
show_train_history(history, "loss", "val_loss")

scores = model.evaluate(x_test_normal, y_test_onehot)
print("accuracy = ", scores[1])

prediction = model.predict_classes(x_test_normal)
print("prediction: ", prediction[:10])

prediction_prob = model.predict(x_test_normal)

plot_image_label_prediction(x_test_image, y_test_label, prediction, idx=0, num=25)
show_prediction_prob(y_test_label, prediction, x_test_image, prediction_prob, 0)

print(pd.crosstab(y_test_label.reshape(-1), prediction, rownames = ["label"], colnames = ["predict"]))
