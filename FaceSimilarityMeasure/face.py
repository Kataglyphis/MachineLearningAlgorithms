import os
from os import listdir
from os.path import isfile, join

from argparse import ArgumentParser
import time
import random

from numpy.core.fromnumeric import shape

import cv2
import numpy as np

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#keras stuff
from tensorflow import keras
#import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
#from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, merge, Dropout, BatchNormalization, InputLayer, concatenate, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Subtract, GlobalAveragePooling2D
#from keras.layers import Conv2D, MaxPooling2D, LocallyConnected2D
#from keras.layers.core import *
#from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras import regularizers
#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.optimizers import SGD
#from keras_vggface.vggface import VGGFace
#from keras_vggface import utils
from keras.initializers import Initializer


#from sklearn.metrics import roc_auc_score
#def auroc(y_true, y_pred):
#    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

#from tensorflow.python.client import device_lib
import tensorflow as tf

from matplotlib import pyplot as plt


# size of the images; shape =(size,size)
SIZE=224

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

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

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
    seq = Sequential([

        Conv2D(64,(10,10), input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(2e-4), kernel_initializer=initialize_weights()),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128,(7,7), input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(2e-4), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128,(4,4), input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(2e-4), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(256,(4,4), input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(2e-4), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),

        Flatten(),
        #Dense(512, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-3), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        #Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-3), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        Dense(512, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-3), kernel_initializer=initialize_weights(), bias_initializer=initialize_bias()),
        Dropout(0.2),

        # InputLayer(input_shape=input_shape),
        # Conv2D(64,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # Conv2D(64,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # #Dropout(0.2),

        # MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        
        # Conv2D(128,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # Conv2D(128,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # #Dropout(0.2),

        # MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        
        # Conv2D(256,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # Conv2D(256,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # Conv2D(256,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # #Dropout(0.2),

        # MaxPooling2D(pool_size=(2,2), strides=(2,2)),

        # Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # #Dropout(0.2),

        # MaxPooling2D(pool_size=(2,2), strides=(2,2)),

        # Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # #Dropout(0.2),

        # MaxPooling2D(pool_size=(2,2), strides=(2,2)),


        # Flatten(),
        # Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # #Dropout(0.2),
        # Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        # #Dropout(0.2),

    ])
    print(seq.summary())
    return seq

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

    
    # prepare our left and right input of siamese network
    # positives
    left_input_train_positive = []
    right_input_train_positive = []
    left_input_val_positive = []
    right_input_val_positive = []

    # negatives
    left_input_train_negative = []
    right_input_train_negative = []
    left_input_val_negative = []
    right_input_val_negative = []

    # initialize training data
    positive_train_pairs = create_positive_pairs(x_train)
    positive_val_pairs = create_positive_pairs(x_val)

    # print('Negative training pairs #' + str(len(positive_train_pairs)))

    for training_pair in positive_train_pairs:
        left_input_train_positive.append(training_pair[0])
        right_input_train_positive.append(training_pair[1])

    for training_pair in positive_val_pairs:
        left_input_val_positive.append(training_pair[0])
        right_input_val_positive.append(training_pair[1])

    negative_train_pairs = create_negative_pairs(x_train)
    negative_val_pairs = create_negative_pairs(x_val)

    # print('Negative training pairs #' + str(len(negative_train_pairs)))

    for training_pair in negative_train_pairs:
        left_input_train_negative.append(training_pair[0])
        right_input_train_negative.append(training_pair[1])

    for training_pair in negative_val_pairs:
        left_input_val_negative.append(training_pair[0])
        right_input_val_negative.append(training_pair[1])

    # negatives are standing in the very beginning of the array!!! mind that because of def array.extend()!!
    left_input_train = np.concatenate((np.array(left_input_train_negative), np.array(left_input_train_positive)), axis=0) 
    right_input_train = np.concatenate((np.array(right_input_train_negative), np.array(right_input_train_positive)), axis=0) 
    left_input_val = np.concatenate((np.array(left_input_val_negative), np.array(left_input_val_positive)), axis=0) 
    right_input_val = np.concatenate((np.array(right_input_val_negative), np.array(right_input_val_positive)), axis=0)

    # for x in range(len(left_input_train_positive)):
    #     cv2.imshow("Positive Left", left_input_train_positive[x])
    #     cv2.imshow("Positive Right", right_input_train_positive[x])
    #     cv2.waitKey(0) & 0xFF     

    # for x in range(len(left_input_train_negative)):
    #     cv2.imshow("Negative Left", left_input_train_negative[x])
    #     cv2.imshow("Negative Right", right_input_train_negative[x])
    #     cv2.waitKey(0) & 0xFF  
    # img_rows = x_train.shape[0]
    # img_cols = x_train[1].shape[0]
    input_shape = (SIZE,SIZE,3)

    # cv2.imshow("First Left", left_input_train[0])
    # cv2.imshow("First Right", right_input_train[0])
    # cv2.waitKey(0) & 0xFF  
    # print('Input shape = ' + str(input_shape))
    #from here train our siamese network
    #for each element of the pair create own sequential
    # print("X_Train.shape = " + str(x_train.shape))
    # print("Y_Train.shape = " + str(y_train.shape))
    # print("X_Val.shape = " + str(x_val.shape))
    # print("Y_Test.shape = " + str(y_test.shape))    

    print("X_Train_left.shape = " + str(left_input_train.shape))
    print("X_Val_left.shape = " + str(left_input_val.shape))
    print("X_train_right.shape = " + str(right_input_train.shape))
    print("X_Val_right.shape = " + str(right_input_val.shape))    


    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    #base_network = create_seq_model(input_shape)
    #base_network = create_deep_face_model(input_shape)
    #base_network = VGGFace(model='resnet50', include_top=False, input_shape=input_shape)
    input_tensor = Input(shape=input_shape)
    base_network = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

    for layer in base_network.layers:
        layer.trainable = False

    #for x in base_network.layers[:-3]:
    #   x.trainable = True

    # last_layer = base_network.get_layer('avg_pool').output
    # x = Flatten(name='flatten')(last_layer)
    # out_1 = Dense(4096, activation='sigmoid', name='classifier')(x)
    # out_2 = Dropout(0.1)(out_1)
    # base_network = Model(base_network.input, out_2)
    x = base_network.output
    x = GlobalAveragePooling2D()(x)
    #let's add a fully-connected layer
    x = Dense(4096, activation='sigmoid')(x)
    base_network_v2 = Model(inputs=base_network.input, outputs=x)

    encoded_l = base_network_v2(inputs=left_input)
    encoded_r = base_network_v2(inputs=right_input)

    L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    adam = keras.optimizers.Adam(learning_rate=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=1e-6, amsgrad=False)
    sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9) # decay=1e-6, nesterov=True
    siamese_net.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.001, momentum=0.9),
                metrics=["accuracy"])

    # encoded_l = Concatenate(axis=-1)([GlobalMaxPool2D()(encoded_l), GlobalAvgPool2D()(encoded_l)])
    # encoded_r = Concatenate(axis=-1)([GlobalMaxPool2D()(encoded_r), GlobalAvgPool2D()(encoded_r)])

    # x3 = Subtract()([encoded_l,encoded_r])
    # x3 = Multiply()([x3,x3])

    # x1_ = Multiply()([encoded_l,encoded_l])
    # x2_ = Multiply()([encoded_r,encoded_r])
    # x4 = Subtract()([x1_, x2_])

    # x5 = Lambda(cosine_distance, output_shape = cos_dist_output_shape)([encoded_l,encoded_r])

    # x = Concatenate(axis=-1)([x5,x4,x3])
    # x = Dense(100, activation='sigmoid')(x)
    # x = Dropout(0.01)(x)
    # out = Dense(1, activation='sigmoid')(x)
    #siamese_net = Model(inputs=[left_input,right_input],outputs=out)
    # adam = keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=1e-6, amsgrad=False)
    # sgd = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.99, nesterov=True)
    # siamese_net.compile(loss="binary_crossentropy", optimizer=adam,
    #             metrics=["accuracy"])


    # L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

    # L1_distance = L1_layer([encoded_l, encoded_r])

    # prediction = Dense(1,activation='sigmoid', bias_initializer=initialize_bias())(L1_distance)
    # siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # adam = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=1e-6, amsgrad=False)
    # sgd = keras.optimizers.SGD(lr=0.01, momentum=0.95, nesterov=True) # decay=1e-6
    # siamese_net.compile(loss="binary_crossentropy", optimizer=sgd,
    #             metrics=["accuracy"])

    print(siamese_net.summary())
    #datagen.fit(x_train)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    #,callbacks=[callback]
    history = siamese_net.fit(
        datagen.flow([left_input_train, right_input_train], 
        y_train, batch_size=batch_size),
        epochs=epochs,
        verbose = 1,

        validation_data=(datagen.flow([left_input_val, right_input_val], y_test, batch_size=batch_size)))

    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show(block=True)

    directory = './data/plots/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    plt.savefig(directory + 'accuracy.png')

    plt.clf()
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
    siamese_net.save("./data/models/v1.h5")


def test_model():

    input_shape=[]#(SIZE,SIZE,3)
    #input_shape = np.array(input_shape)
    shape=np.array(input_shape)
    #classifier = load_model("./data/models/v1.h5", custom_objects={'initialize_weights': initialize_weights, 
    #                                                                'initialize_bias' : initialize_bias})
    classifier = tf.keras.models.load_model("./data/models/v1.h5")

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

        pred = classifier.predict([image1,image2])[0]

        pred = pred[0]
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

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

# print(model.summary())

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)