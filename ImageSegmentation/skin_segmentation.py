import os
from os import listdir
from os.path import isfile, join
import sys

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
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import concatenate
from keras.layers.core import Lambda

#from tensorflow.python.client import device_lib
import tensorflow as tf

from keras import backend as K

from matplotlib import pyplot as plt


# size of the images; shape =(size,size)
SIZE=480


def load_images(path, load_mask_images):
    images = []
    for f in listdir(path):
        if isfile(join(path, f)):
            if load_mask_images and (f.startswith('mask') is True):

                image = cv2.imread(path + f, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                images.append(image)

            if (load_mask_images is False) and (f.startswith('mask') is False):
                
                image = cv2.imread(path + f)
                image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                images.append(image)

    return images

def prepare_data(): 
    
    training_images = []
    mask_training_images = []
    validation_images = []
    mask_validation_images = []

    directory_train = "./data/train/"
    directory_validation = "./data/validation/"

    training_images = load_images(directory_train, False)

    print('Loaded training images!')

    mask_training_images = load_images(directory_train, True)

    print('Loaded training masks!')

    validation_images = load_images(directory_validation, False)
    
    print('Loaded validation images!')

    mask_validation_images = load_images(directory_validation, True)  

    print('Loaded validation masks!')

    directory = './data/npz/'
    if not os.path.exists(directory):
        os.mkdir(directory)

    np.savez('./data/npz/training_images.npz', np.array(training_images))
    np.savez('./data/npz/mask_training_images.npz', np.array(mask_training_images))
    np.savez('./data/npz/validation_images.npz', np.array(validation_images))
    np.savez('./data/npz/mask_validation_images.npz', np.array(mask_validation_images))

    print('Saved images!')

    return

def load_data_training_and_test():
    
    directory = './data/npz/'
    # if not os.path.exists(directory):
    #     os.mknod(directory)

    npzfile = np.load(directory + 'training_images.npz')
    train = npzfile['arr_0']

    npzfile = np.load(directory + 'mask_training_images.npz')
    train_masks = npzfile['arr_0']

    npzfile = np.load(directory + 'validation_images.npz')
    validation = npzfile['arr_0']

    npzfile = np.load(directory + 'mask_validation_images.npz')
    validation_masks = npzfile['arr_0']

    return (train, train_masks), (validation, validation_masks)

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def train(batch_size, epochs):
    

    (x_train, m_train), (x_test, m_test) = load_data_training_and_test()

    print("Train masks shape before is: " + str(m_train.shape))
    print("Test masks shape before is: " + str(m_test.shape))

    m_train = m_train[:, :, :,np.newaxis]
    m_test = m_test[: , :, :, np.newaxis]

    m_train = m_train.astype('float32')
    m_test = m_test.astype('float32')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # x_train /= 255
    # x_test /= 255
    m_train /= 255
    m_test /= 255

    img_rows = x_train[0].shape[0]
    img_cols = x_train[1].shape[0]
    input_shape = (img_rows,img_cols,3)

    print("Train shape is: " + str(x_train.shape))
    print("Test shape is: " + str(x_test.shape))
    print("Train masks shape is: " + str(m_train.shape))
    print("Test masks shape is: " + str(m_test.shape))

    # shaping the UNet
    inputs = Input((SIZE,SIZE, 3))
    s = Lambda(lambda x:x/255)(inputs)

    c1 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    d1 = Dropout(0.1)(c1)
    c2 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d1)
    p1 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    d2 = Dropout(0.1)(c3)
    c4 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d2)
    p2 = MaxPooling2D((2,2))(c4)

    c5 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    d3 = Dropout(0.2)(c5)
    c6 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d3)
    p3 = MaxPooling2D((2,2))(c6)

    c7 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    d4 = Dropout(0.2)(c7)
    c8 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d4)
    p4 = MaxPooling2D((2,2))(c8)

    c9 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    d5 = Dropout(0.3)(c9)
    c10 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d5)
    
    t1 = Conv2DTranspose(128,(2,2), strides=(2,2), padding='same')(c10)
    co1 = concatenate([t1, c8], axis=3)
    c11 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(co1)
    d6 = Dropout(0.2)(c11)
    c12 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d6)

    t2 = Conv2DTranspose(64,(2,2), strides=(2,2), padding='same')(c12)
    co2 = concatenate([t2, c6], axis=3)
    c13 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(co2)
    d7 = Dropout(0.2)(c13)
    c14 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d7)

    t3 = Conv2DTranspose(32,(2,2), strides=(2,2), padding='same')(c14)
    co3 = concatenate([t3, c4], axis=3)
    c15 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(co3)
    d8 = Dropout(0.1)(c15)
    c16 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d8)

    t4 = Conv2DTranspose(16,(2,2), strides=(2,2), padding='same')(c16)
    co4 = concatenate([t4, c2], axis=3)
    c17 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(co4)
    d9 = Dropout(0.1)(c17)
    c18 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d9)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(c18)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
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

    history = model.fit(x_train, m_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, m_test),
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

    directory = './data/out/'
    if not os.path.exists(directory):
        os.mkdir(directory)

    test_files = './data/test/'
    prefix = 'out-'

    for f in listdir(test_files):
        print(f)
        if isfile(join(test_files, f)):
            image = cv2.imread(test_files + f)
            image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            image = image.reshape(1, SIZE, SIZE, 3)
            if image is None:
                print("Image not loaded!")

            pred = classifier.predict(image, 1, verbose = 1)[0]
            pred = pred * 255
            #predictions_test = (pred > 0.5).astype(np.uint8)
            predictions_test = (pred).astype(np.uint8)


            skin_pred = predictions_test

            save_dir = directory + prefix + f
            skin_pred = cv2.resize(skin_pred, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imwrite(save_dir, skin_pred)

    # for file in test_files:
    #     file = file[:-1]
    #     print(file)
        #directory = './data/test/'
        
        # image = cv2.imread(test_files + '/' + file)
        # image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        # image = image.reshape(1, SIZE, SIZE, 3)
        # if image is None:
        #     print("Image not loaded!")

        # pred = classifier.predict(image, 1, verbose = 1)[0]

        # predictions_test = (pred > 0.5).astype(np.uint8)

        # skin_pred = predictions_test

        # save_dir = directory + prefix + file
        # cv2.imwrite(save_dir, skin_pred)
        

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