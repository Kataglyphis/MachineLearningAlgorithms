import os
from os import listdir
from os.path import isfile, join

from argparse import ArgumentParser

import cv2
import numpy as np

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#keras stuff
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D, MaxPool2D
from keras.applications.vgg16 import VGG16

#from tensorflow.python.client import device_lib
import tensorflow as tf

from keras import backend as K

import matplotlib.pyplot as plt

# size of the images; shape =(size,size)
SIZE=150


def load_images_and_labels(path,label):
    images = []
    labels = []
    for f in listdir(path):
        if isfile(join(path, f)):
            image = cv2.imread(path + f)
            image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            images.append(image)
            labels.append(label)

    return images, labels

def prepare_data(): 
    
    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []

    directory_train_p = "./data/train/p/"
    directory_train_n = "./data/train/n/"
    directory_validation_p = "./data/validation/p/"
    directory_validation_n = "./data/validation/n/"

    training_images_p, training_labels_p = load_images_and_labels(directory_train_p, 1)
    training_images = np.array(training_images_p)
    training_labels = np.array(training_labels_p)

    print('Loaded p training images!')

    training_images_n, training_labels_n = load_images_and_labels(directory_train_n, 0)
    training_images = np.concatenate((training_images,training_images_n))
    training_labels = np.concatenate((training_labels, training_labels_n))

    print('Loaded n training images!')

    validation_images_p, validation_labels_p = load_images_and_labels(directory_validation_p, 1)
    validation_images = np.array(validation_images_p)
    validation_labels = np.array(validation_labels_p)

    print('Loaded p valid images!')

    validation_images_n, validation_labels_n = load_images_and_labels(directory_validation_n, 0)
    validation_images = np.concatenate((validation_images,validation_images_n))
    validation_labels = np.concatenate((validation_labels,validation_labels_n))

    print('Loaded n valid images!')

    directory = './data/npz/'
    if not os.path.exists(directory):
        os.mkdir(directory)

    np.savez('./data/npz/training_data.npz', np.array(training_images))
    np.savez('./data/npz/training_labels.npz', np.array(training_labels))
    np.savez('./data/npz/validation_data.npz', np.array(validation_images))
    np.savez('./data/npz/validation_labels.npz', np.array(validation_labels))

    print('Saved images!')

    return

def load_data_training_and_test():
    
    directory = './data/npz/'
    # if not os.path.exists(directory):
    #     os.mknod(directory)

    npzfile = np.load(directory + 'training_data.npz')
    train = npzfile['arr_0']

    npzfile = np.load(directory + 'training_labels.npz')
    train_labels = npzfile['arr_0']

    npzfile = np.load(directory + 'validation_data.npz')
    validation = npzfile['arr_0']

    npzfile = np.load(directory + 'validation_labels.npz')
    validation_labels = npzfile['arr_0']

    return (train, train_labels), (validation, validation_labels)


def train(batch_size, epochs):

    (x_train, y_train), (x_test, y_test) = load_data_training_and_test()

    print("y train shape before:" + str(y_train.shape))
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print("y train shape after:" + str(y_train.shape))


    x_train /= 255
    x_test /= 255

    #img_rows = x_train[0].shape[0]
    #img_cols = x_train[1].shape[0]
    #input_shape = (img_rows,img_cols,3)
    input_shape = (SIZE,SIZE,3)

    print("X_Train.shape = " + str(x_train.shape))
    print("Y_Train.shape = " + str(y_train.shape))
    print("X_Val.shape = " + str(x_test.shape))
    print("Y_Test.shape = " + str(y_test.shape)) 

    model = Sequential()

    model.add(Conv2D(input_shape=(SIZE,SIZE,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=1, activation="softmax"))
    # model.add(Conv2D(16, kernel_size=(3,3),input_shape=input_shape, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))

    # model.add(Conv2D(32, kernel_size=(3,3),input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Conv2D(32, (3,3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Conv2D(64, (3,3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    # train_datagen = ImageDataGenerator(
    #     rescale = 1./255,
    #     shear_range = 0.2,
    #     zoom_range = 0.2,
    #     horizontal_flip = False)

    # validation_datagen = ImageDataGenerator(rescale = 1./255)

    # train_generator = train_datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size = (size, size),
    #     batch_size = batch_size,
    #     class_mode = 'binary')
    
    # validation_generator = validation_datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(size,size),
    #     batch_size=batch_size,
    #     class_mode='binary')

    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch = nb_train_samples // batch_size,
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps= nb_validation_samples // batch_size)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show(block=True)

    directory = './data/plots/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    plt.savefig(directory + 'accuracy.png')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show(block=True)

    plt.savefig(directory + 'loss.png')

    directory = './data/models/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    model.save("./data/models/v1.h5")


def test_model():

    classifier = load_model("./data/models/v1.h5")

    test_files = open("./data/testFiles.txt", "r")
    #if os.path.isfile('./data/result.txt') is None:
    os.remove('./data/result.txt')
    result = open('./data/result.txt','w')

    for file in test_files:
        file = file[:-1]
        print(file)
        directory = './data/test'
        
        image = cv2.imread(directory + "/" + file)
        image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        image = image.reshape(1, SIZE, SIZE, 3)
        if image is None:
            print("Image not loaded!")

        pred = classifier.predict(image, 1, verbose = 0)[0]

        pred = pred[0] - 0.5

        result.write(str(pred) + '\n')


    test_files.close()
    result.close()

parser = ArgumentParser(description="Control the learning process")
parser.add_argument("-c", "--command", required=True, help='the process we want to execute in order of the NN')
parser.add_argument("-b", "--batch_size", type=int, help='# of batches processed in each step')
parser.add_argument("-e", "--epochs", type=int, help='# of epochs we run the training process on')
parser.add_argument("-m", "--mode", help='switch between CPU and GPU')



args = parser.parse_args()
# print (args.command)
if args.mode == 'GPU':

    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)

if args.command == 'train':

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # K.tensorflow_backend._get_available_gpus()
    train(args.batch_size, args.epochs)

elif args.command == 'prepare_data':

    prepare_data()

elif args.command == 'test':  

    test_model()

print("Finished")
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

# print(model.summary())

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)