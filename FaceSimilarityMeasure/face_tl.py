from math import sqrt
import os
from os import listdir
from os.path import isfile, join

from argparse import ArgumentParser
import time
import random

from numpy.core.fromnumeric import shape

import cv2
import numpy as np
from numpy.linalg import norm

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#keras stuff
#from tensorflow import keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, merge, Dropout, BatchNormalization, InputLayer, concatenate, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Subtract, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected2D
from keras.layers.core import *
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras import regularizers
#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, Concatenate, GlobalMaxPool2D, GlobalAvgPool2D
#from tensorflow.keras.optimizers import SGD
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.initializers import Initializer


#from sklearn.metrics import roc_auc_score
#def auroc(y_true, y_pred):
#    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

#from tensorflow.python.client import device_lib
import tensorflow as tf

from matplotlib import pyplot as plt


# size of the images; shape =(size,size)
SIZE=224

# scale the number of triplets
SCALE_TRIPLETS= 20

class initialize_weights( Initializer ):
    
  def __call__(self, shape, dtype=None):

    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)


class initialize_bias( Initializer ):
  
  def __call__(self, shape, dtype=None):

    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

# def initialize_weights(shape, dtype=None):
#     """
#         The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
#         suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
#     """
#     return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

# def initialize_bias(shape, dtype=None):
#     """
#         The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
#         suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
#     """
#     return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def cosine_distance(a,b):

    similarity = 0
    for x,y in zip(a,b):
        similarity += abs(x-y) ** 2
    
    similarity = sqrt(similarity)
    return 1 / similarity
    # return np.dot(a,b) /(norm(a) * norm(b))

def load_images(path,images):
    for f in sorted(listdir(path)):
        # print(f)
        if isfile(join(path, f)):
            image = cv2.imread(path + f)
            image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            images.append(image)
            if image is None:
                print("loaded image not correctly!")
            #print('Shape of images array: ' + str(np.array(images).shape))
    return images

def create_positive_pairs(training_images):
    result = []
    length = len(training_images)
    i = 0
    while i < (length - 1):
        #positive examples lay in sorted order in our data folder
        pair=[]
        pair.append(training_images[i])
        pair.append(training_images[i+1])
        # cv2.imshow("Positive First", training_images[i])
        # cv2.imshow("Positive Second", training_images[i+1])
        # cv2.waitKey(0) & 0xFF
        result.append(pair)
        i += 2
    # print('Shape of result array is ' + str(np.array(result).shape))
    # print('Created #' + str(len(result)) + ' of positive pairs')
    return result

def create_negative_pairs(training_images):

    random.seed(int(round(time.time() * 1000)))
    result = []
    length = len(training_images)
    i = 0
    while i < (length - 1):
        negative_pair=[]
        while(True):
            false_sec_pair_index = int(random.random() * (length - 1))
            if(false_sec_pair_index != i and false_sec_pair_index != i+1):
                break
        # print('index i is ' + str(i) + ' and random index to it is ' + str(false_sec_pair_index))
        #positive examples lay in sorted order in our data folder
        negative_pair.append(training_images[i])
        negative_pair.append(training_images[false_sec_pair_index])
        # cv2.imshow("Negative First", training_images[i])
        # cv2.imshow("Negative Second", training_images[false_sec_pair_index])
        # cv2.waitKey(0) & 0xFF
        result.append(negative_pair)
        i += 2
    # print('Shape of result array is ' + str(np.array(result).shape))
    # print('Created #' + str(len(result)) + ' of negative pairs')
    return result

def create_triplets(images):
    
    datagen = ImageDataGenerator(
        rotation_range=45,
        shear_range=0.2,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=None,
    )

    anchors = []
    positives = []
    negatives = []

    length = len(images)
    i=0
    while i < (length - 1):
        #triplet = []
        #triplet.append(images[i])
        #triplet.append(images[i+1])
        j = 0

        while j < SCALE_TRIPLETS: 

            #new_image_first = datagen.flow([images[i]])
            #new_image_second = datagen.flow([images[i+1]])      
            anchors.append(images[i])
            positives.append(images[i+1])
            while(True):
                false_sec_pair_index = int(random.random() * (length - 1))
                if(false_sec_pair_index != i and false_sec_pair_index != i+1):
                    break
                # print('index i is ' + str(i) + ' and random index to it is ' + str(false_sec_pair_index))
                #positive examples lay in sorted order in our data folder
                
            negatives.append(images[false_sec_pair_index])
            j+=1
            # cv2.imshow("Negative First", training_images[i])
            # cv2.imshow("Negative Second", training_images[false_sec_pair_index])
            # cv2.waitKey(0) & 0xFF
        i+=2

    return anchors, positives, negatives

def prepare_data(): 

    training_images = []
    validation_images = []

    directory_train = "./data/train/"
    directory_validation = "./data/validation/"

    training_images = load_images(directory_train, training_images)

    validation_images = load_images(directory_validation, validation_images)

    # p_pairs_training = create_positive_pairs(training_images)
    # p_pairs_validation = create_positive_pairs(validation_images)

    # n_pairs_training = create_negative_pairs(training_images)
    # n_pairs_validation = create_negative_pairs(validation_images)

    # training_images = p_pairs_training + n_pairs_training
    # validation_images = p_pairs_validation + n_pairs_validation

    directory = './data/npz/'
    if not os.path.exists(directory):
        os.mkdir(directory)

    np.savez('./data/npz/training_data.npz', np.array(training_images))
    np.savez('./data/npz/validation_data.npz', np.array(validation_images))

    print('Successfully prepared data! :)')

    return

def load_data_training_and_test():
    
    directory = './data/npz/'
    # if not os.path.exists(directory):
    #     os.mknod(directory)

    npzfile = np.load(directory + 'training_data.npz')
    train = npzfile['arr_0']

    npzfile = np.load(directory + 'validation_data.npz')
    validation = npzfile['arr_0']

    return train, validation

def create_deep_face_model(input_shape):

    base_model = Sequential()
    base_model.add(Conv2D(32, (11, 11), activation='relu', name='C1', input_shape=input_shape))
    base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
    base_model.add(Conv2D(16, (9, 9), activation='relu', name='C3'))
    base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
    base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5') )
    base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
    base_model.add(Flatten(name='F0'))
    base_model.add(Dense(4096, activation='sigmoid', name='F7'))
    base_model.add(Dropout(rate=0.5, name='D0'))
    return base_model

def create_seq_model(input_shape):

    # seq = Sequential([
    #     Conv2D(32,(4,4), input_shape=input_shape, activation='relu'),
    #     Conv2D(32,(4,4), input_shape=input_shape, activation='relu'),
    #     MaxPooling2D((3,3)),
    #     Conv2D(64,(3,3), input_shape=input_shape, activation='relu'),
    #     Conv2D(64,(3,3), input_shape=input_shape, activation='relu'),
    #     MaxPooling2D((2,2)),
    #     Flatten(),
    #     Dense(128, activation='sigmoid'),
    #     # Dropout(rate=0.5),
    #     # Dense(8631, activation='softmax')
    # ])

    # https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    base_model = Sequential([

        Conv2D(64,(10,10), input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(2e-4), kernel_initializer=initialize_weights()),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128,(7,7), activation='relu', kernel_regularizer=regularizers.l2(2e-4), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128,(4,4), activation='relu', kernel_regularizer=regularizers.l2(2e-4), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(256,(4,4), activation='relu', kernel_regularizer=regularizers.l2(2e-4), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        BatchNormalization(),

        Flatten(),
        #Dense(512, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-3), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        #Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-3), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        Dense(512, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-3), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        Dropout(0.5),

    ])
    print(base_model.summary())

    return base_model

def base_model_for_triplet_loss():

    input_shape=(SIZE,SIZE,3)
    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)

    base_model = VGGFace(model='resnet50', include_top=False, input_shape=input_shape)

    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(anchor_input)
    x2 = base_model(positive_input)
    x3 = base_model(negative_input)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])
    x3 = Concatenate(axis=-1)([GlobalMaxPool2D()(x3), GlobalAvgPool2D()(x3)])

    
    concat_vector = concatenate([x1, x2, x3], axis=-1, name='concat')
    
    siamese_net = Model([anchor_input, positive_input, negative_input], concat_vector)

    #adam = keras.optimizers.Adam(learning_rate=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=1e-6, amsgrad=False)
    #sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9) # decay=1e-6, nesterov=True
    siamese_net.compile(loss=triplet_loss, optimizer=SGD(lr=0.1, momentum=0.9))

    return siamese_net

def triplet_loss(y_true, y_pred, alpha = 0.4):

    total_length = y_pred.shape.as_list()[-1]

    anchor = y_pred[:,0:int(total_length * 1/3)]
    positive = y_pred[:,int(total_length * 1/3):int(total_length * 2/3)]
    negative = y_pred[:,int(total_length * 2/3):int(total_length)]

    pos_dist = K.sum(K.square(anchor-positive), axis=1)

    neg_dist = K.sum(K.square(anchor-negative), axis=1)

    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

def train(batch_size, epochs):

    # data is stored in pairs; first half being p; rest n
    x_train, x_val= load_data_training_and_test()

    datagen = ImageDataGenerator(
        rotation_range=45,
        shear_range=0.2,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=None,
    )
    # print('x_train shape is ' + str(x_train.shape))
    # print('x_train_shape[0] is ' + str(x_train.shape[0]))

    # create appropriate labels, 0 for p, 1 for n
    y_train = np.full(int(x_train.shape[0] / 2), 1)
    y_train_n = np.full(int(x_train.shape[0] / 2), 0)
    y_train = np.concatenate((y_train_n,y_train), axis=0)

    y_test = np.full(int(x_val.shape[0] / 2), 1)
    y_test_n = np.full(int(x_val.shape[0] / 2), 0)
    y_test = np.concatenate((y_test_n,y_test), axis=0)

    # we want to train on normalized inputs
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_val /= 255

    anchors_train, positives_train, negatives_train = create_triplets(x_train)
    anchors_val, positives_val, negatives_val = create_triplets(x_val)


    y_train = np.full(int(x_train.shape[0] / 2) * SCALE_TRIPLETS, 0)
    y_test = np.full(int(x_val.shape[0] / 2)* SCALE_TRIPLETS, 0)
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    
    input_shape = (SIZE,SIZE,3)

    base_model = create_seq_model(input_shape)#base_model_for_triplet_loss()

    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)

    x1 = base_model(anchor_input)
    x2 = base_model(positive_input)
    x3 = base_model(negative_input)

    # x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    # x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])
    # x3 = Concatenate(axis=-1)([GlobalMaxPool2D()(x3), GlobalAvgPool2D()(x3)])

    
    concat_vector = concatenate([x1, x2, x3], axis=-1, name='concat')
    
    siamese_net = Model([anchor_input, positive_input, negative_input], concat_vector)
    siamese_net.compile(loss=triplet_loss, optimizer=SGD(lr=0.0001, momentum=0.9))

    print(siamese_net.summary())

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    history = siamese_net.fit(
        [anchors_train, positives_train, negatives_train], 
        y_train, batch_size=batch_size,
        callbacks=[callback],
        epochs=epochs,
        verbose = 1,
        validation_data=([anchors_val, positives_val, negatives_val], y_test))

    weights_directory = './data/weights/'
    if not os.path.exists(weights_directory):
        os.mkdir(weights_directory)

    weights_filename = weights_directory + "weights.h5"
    siamese_net.save_weights(weights_filename, overwrite=True)

    input_1 = Input(shape=(SIZE, SIZE, 3))
    input_2 = Input(shape=(SIZE, SIZE, 3))
    input_3 = Input(shape=(SIZE, SIZE, 3))

    base_model = create_seq_model(input_shape)

    x1 = base_model(input_1)
    x2 = base_model(input_2)
    #x3 = base_model(input_3)

    concat_vector = concatenate([x1, x2, x3], axis=-1, name='concat')

    trained_model = Model(inputs=(input_1, input_2), outputs=concat_vector)

    trained_model.load_weights(weights_filename)
    #trained_model.compile(loss = triplet_loss, optimizer=sgd)

    print(history.history.keys())
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    #plt.show(block=True)

    directory = './data/plots/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    #plt.savefig(directory + 'accuracy.png')

    #plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylim(0,1)
    #plt.show(block=True)

    plt.savefig(directory + 'loss.png')

    directory = './data/models/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    siamese_net.save("./data/models/v1.h5")
    trained_model.save("./data/models/trained_model.h5")

def retrain(batch_size, epochs):

    # data is stored in pairs; first half being p; rest n
    x_train, x_val= load_data_training_and_test()

    datagen = ImageDataGenerator(
        rotation_range=45,
        shear_range=0.2,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=None,
    )
    # print('x_train shape is ' + str(x_train.shape))
    # print('x_train_shape[0] is ' + str(x_train.shape[0]))

    # create appropriate labels, 0 for p, 1 for n
    y_train = np.full(int(x_train.shape[0] / 2), 1)
    y_train_n = np.full(int(x_train.shape[0] / 2), 0)
    y_train = np.concatenate((y_train_n,y_train), axis=0)

    y_test = np.full(int(x_val.shape[0] / 2), 1)
    y_test_n = np.full(int(x_val.shape[0] / 2), 0)
    y_test = np.concatenate((y_test_n,y_test), axis=0)

    # we want to train on normalized inputs
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_val /= 255

    anchors_train, positives_train, negatives_train = create_triplets(x_train)
    anchors_val, positives_val, negatives_val = create_triplets(x_val)


    y_train = np.full(int(x_train.shape[0] / 2) * SCALE_TRIPLETS, 0)
    y_test = np.full(int(x_val.shape[0] / 2)* SCALE_TRIPLETS, 0)
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    
    input_shape = (SIZE,SIZE,3)

    base_model = create_seq_model(input_shape)#base_model_for_triplet_loss()

    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)

    x1 = base_model(anchor_input)
    x2 = base_model(positive_input)
    x3 = base_model(negative_input)

    # x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    # x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])
    # x3 = Concatenate(axis=-1)([GlobalMaxPool2D()(x3), GlobalAvgPool2D()(x3)])

    
    concat_vector = concatenate([x1, x2, x3], axis=-1, name='concat')
    
    siamese_net = Model([anchor_input, positive_input, negative_input], concat_vector)
    siamese_net.compile(loss=triplet_loss, optimizer=SGD(lr=0.0001, momentum=0.95))

    print(siamese_net.summary())

    weights_directory = './data/weights/'

    weights_filename = weights_directory + "weights.h5"
    siamese_net.load_weights(weights_filename)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    history = siamese_net.fit(
        [anchors_train, positives_train, negatives_train], 
        y_train, batch_size=batch_size,
        callbacks=[callback],
        epochs=epochs,
        verbose = 1,
        validation_data=([anchors_val, positives_val, negatives_val], y_test))

    siamese_net.save_weights(weights_filename, overwrite=True)

    input_1 = Input(shape=(SIZE, SIZE, 3))
    input_2 = Input(shape=(SIZE, SIZE, 3))
    #input_3 = Input(shape=(SIZE, SIZE, 3))

    base_model = create_seq_model(input_shape)

    x1 = base_model(input_1)
    x2 = base_model(input_2)
    #x3 = base_model(input_3)

    concat_vector = concatenate([x1, x2], axis=-1, name='concat')

    trained_model = Model(inputs=(input_1, input_2), outputs=concat_vector)

    trained_model.load_weights(weights_filename)
    #trained_model.compile(loss = triplet_loss, optimizer=sgd)

    print(history.history.keys())
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    #plt.show(block=True)

    directory = './data/plots/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    #plt.savefig(directory + 'accuracy.png')

    #plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylim(0,1)
    #plt.show(block=True)

    plt.savefig(directory + 'loss.png')

    directory = './data/models/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    siamese_net.save("./data/models/v1.h5")
    trained_model.save("./data/models/trained_model.h5")

def test_model():

    input_shape=[]#(SIZE,SIZE,3)
    #input_shape = np.array(input_shape)
    shape=np.array(input_shape)
    classifier = load_model("./data/models/trained_model.h5", custom_objects={'initialize_weights': initialize_weights, 
                                                                   'initialize_bias' : initialize_bias,
                                                                   'triplet_loss' : triplet_loss})
    #classifier = tf.keras.models.load_model("./data/models/trained_model.h5")
    # weights_directory = './data/weights/'
    # weights_filename = weights_directory + "weights.hdf5"
    # classifier.load_weights(weights_filename)

    test_files = open("./data/testPairs.txt", "r")

    if os.path.isfile('./data/result.txt'):
        os.remove('./data/result.txt')
    result = open('./data/result.txt','w')

    for file in test_files:
        file = file[:-1]
        #print(file)
        file1, file2 = file.split() 
        directory = './data/test'
        
        image1 = cv2.imread(directory + "/" + file1)
        image1 = cv2.resize(image1, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imshow("Image1", image1)
        image1 = image1.reshape(1, SIZE, SIZE, 3)
        if image1 is None:
            print("Image not loaded!")

        image2 = cv2.imread(directory + "/" + file2)
        image2 = cv2.resize(image2, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imshow("Image2", image2)
        image2 = image2.reshape(1, SIZE, SIZE, 3)
        if image2 is None:
            print("Image not loaded!")

        pred = np.array(classifier.predict([image1,image2]).ravel().tolist())

        #pred = pred[0]
        pred = cosine_distance(pred[:len(pred)//2], pred[len(pred)//2:])
        print('pred = ' + str(pred))
        cv2.waitKey(0) & 0xFF

        #pred = pred

        result.write(str(pred) + '\n')


    test_files.close()
    result.close()
    print('Finished testing :)')

parser = ArgumentParser(description="Control the learning process")
parser.add_argument("-c", "--command", required=True, help='the process we want to execute in order of the NN')
parser.add_argument("-b", "--batch_size", type=int, help='# of batches processed in each step')
parser.add_argument("-e", "--epochs", type=int, help='# of epochs we run the training process on')
parser.add_argument("-m", "--mode", help='switch between CPU and GPU')



args = parser.parse_args()
# print (args.command)
# if args.mode == 'GPU':

#     gpu = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(gpu[0], True)

if args.command == 'train':

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # K.tensorflow_backend._get_available_gpus()
    train(args.batch_size, args.epochs)

elif args.command == 'prepare_data':

    prepare_data()

elif args.command == 'test':  

    test_model()

elif args.command == 'retrain':

    retrain(args.batch_size, args.epochs)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

# print(model.summary())

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)