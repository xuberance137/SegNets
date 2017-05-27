from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential, model_from_json
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K

import numpy as np
import json
import h5py
import argparse
from config import *

np.random.seed(007) # starting seed for reproducibility

def parse_and_set_arguments():
    global TRAINING
    parser = argparse.ArgumentParser(description='Basic Segnet Model Training and Test App')
    parser.add_argument('--train', type=int, action='store', dest='TRAINING', default=1, help='Run Segnet in train or test mode(default is test)')
    args = parser.parse_args()
    TRAINING = args.TRAINING

def create_model():
    model = Sequential()
    # encoding layers
    model.add(Permute((2,3,1), input_shape=(IMAGE_DATA_SHAPE[2],IMAGE_DATA_SHAPE[0],IMAGE_DATA_SHAPE[1])))
    model.add(Convolution2D(FILTER_SIZE, kernel_size= KERNEL_SIZE, padding='same', input_shape=IMAGE_DATA_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE, padding='valid'))

    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(2*FILTER_SIZE, kernel_size= KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(4*FILTER_SIZE, kernel_size= KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(8*FILTER_SIZE, kernel_size= KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # decoding layers
    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(8*FILTER_SIZE, kernel_size= KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=POOL_SIZE))
    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(4*FILTER_SIZE, kernel_size= KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=POOL_SIZE))
    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(2*FILTER_SIZE, kernel_size= KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=POOL_SIZE))
    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(FILTER_SIZE, kernel_size= KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())

    model.add(Convolution2D(CATEGORIES, kernel_size=(1,1), padding='valid'))
    # model.add(Reshape((12,360*480), input_shape=(12,360,480)))
    model.add(Reshape((CATEGORIES, IMAGE_DATA_SHAPE[0]*IMAGE_DATA_SHAPE[1]), input_shape=(CATEGORIES, IMAGE_DATA_SHAPE[0], IMAGE_DATA_SHAPE[1])))
    model.add(Permute((2,1))) #, input_shape=(12,360*480)))
    model.add(Activation('softmax'))

    model.summary()

    with open(MODEL_PATH+'basic_model.json', 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

def train_model():
    # load model architecture
    with open(MODEL_PATH+'basic_model.json', 'r') as model_file:
        segnet_basic_model = model_from_json(model_file.read())

    # load training and test data
    with h5py.File(DATA_PATH+DATASET_FILE, 'r') as hf:
        train_data = hf['train_data'][:]
        train_label = hf['train_label'][:]
        test_data = hf['test_data'][:]
        test_label = hf['test_label'][:]
        val_data = hf['val_data'][:]
        val_label = hf['val_label'][:]

    # defining checkpoints
    model_checkpoint = ModelCheckpoint(MODEL_PATH+'basic_model_{epoch:03d}.hdf5')
    csv_log = CSVLogger(MODEL_PATH+'basic_model_training_log.csv', separator=',', append=False)
    callbacks = [model_checkpoint, csv_log]

    print('Compling Model')
    segnet_basic_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Training Model')
    history = segnet_basic_model.fit(train_data, 
                                    train_label, 
                                    callbacks=callbacks, 
                                    batch_size=BATCH_SIZE, 
                                    epochs=NUM_EPOCH, 
                                    verbose=1,
                                    # class_weight= CLASS_WEIGHTING,
                                    validation_data=(val_data, val_label), #val split=1/3
                                    shuffle=True)


if __name__ == '__main__':
    parse_and_set_arguments()
    create_model()
    #train_model()





