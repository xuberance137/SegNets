from keras.models import Sequential, model_from_json, Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K

import warnings
import numpy as np
import json
import h5py
import argparse
import cv2
from config import *

np.random.seed(007) # starting seed for reproducibility

def parse_and_set_arguments():
    global TRAINING
    global MODEL_TYPE
    parser = argparse.ArgumentParser(description='Basic Segnet Model Training and Test App')
    parser.add_argument('--train', type=int, action='store', dest='TRAINING', default=1, help='Run Segnet in train or test mode(default is test)')
    parser.add_argument('--model', type=int, action='store', dest='MODEL_TYPE', default=0, help='Run Segnet using functional or model API(default is functional)')
    args = parser.parse_args()
    TRAINING = args.TRAINING
    MODEL_TYPE = args.MODEL_TYPE

def _obtain_input_shape(input_shape, default_size, min_size, data_format):
    """Internal utility to compute/validate an ImageNet model's input shape.
    # Arguments
        input_shape: either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: default input width/height for the model.
        min_size: minimum input width/height accepted by the model.
        data_format: image data format to use.
        include_top: whether the model is expected to
            be linked to a classifier via a Flatten layer.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: in case of invalid argument values.
    """
    if data_format == 'channels_first':
        default_shape = (3, default_size, default_size)
    else:
        default_shape = (default_size, default_size, 3)

    if data_format == 'channels_first':
        if input_shape is not None:
            if len(input_shape) != 3:
                raise ValueError('`input_shape` must be a tuple of three integers.')
            if input_shape[0] != 3:
                raise ValueError('The input must have 3 channels; got `input_shape=' + str(input_shape) + '`')
            if ((input_shape[1] is not None and input_shape[1] < min_size) or
               (input_shape[2] is not None and input_shape[2] < min_size)):
                raise ValueError('Input size must be at least ' + str(min_size) + 'x' + str(min_size) + ', got `input_shape=' + str(input_shape) + '`')
        else:
            input_shape = (3, None, None)
    else:
        if input_shape is not None:
            if len(input_shape) != 3:
                raise ValueError('`input_shape` must be a tuple of three integers.')
            if input_shape[-1] != 3:
                raise ValueError('The input must have 3 channels; got `input_shape=' + str(input_shape) + '`')
            if ((input_shape[0] is not None and input_shape[0] < min_size) or
               (input_shape[1] is not None and input_shape[1] < min_size)):
                raise ValueError('Input size must be at least ' + str(min_size) + 'x' + str(min_size) + ', got `input_shape=' + str(input_shape) + '`')
        else:
            input_shape = (None, None, 3)
    
    return input_shape

def create_model_func(input_shape=None):
    """Instantiates the SegNet architecture using the Functional API.
    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels, and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid input shape.
    """
    print('Creating Model using Functional API')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=48, data_format=K.image_data_format())
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=POOL_SIZE, padding='valid')(x)
    
    x = ZeroPadding2D(padding=PAD_SIZE)(x)
    x = Conv2D(128, (3, 3), padding='valid', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=POOL_SIZE, padding='valid')(x)

    x = ZeroPadding2D(padding=PAD_SIZE)(x)
    x = Conv2D(256, (3, 3), padding='valid', name='block1_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=POOL_SIZE, padding='valid')(x)

    x = ZeroPadding2D(padding=PAD_SIZE)(x)
    x = Conv2D(512, (3, 3), padding='valid', name='block1_conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=POOL_SIZE, padding='valid')(x)

    x = ZeroPadding2D(padding=PAD_SIZE)(x)
    x = Conv2D(512, (3, 3), padding='valid', name='block1_conv5')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=POOL_SIZE)(x)
    x = ZeroPadding2D(padding=PAD_SIZE)(x)
    x = Conv2D(256, (3, 3), padding='valid', name='block1_conv6')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=POOL_SIZE)(x)
    x = ZeroPadding2D(padding=PAD_SIZE)(x)
    x = Conv2D(128, (3, 3), padding='valid', name='block1_conv7')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=POOL_SIZE)(x)
    x = ZeroPadding2D(padding=PAD_SIZE)(x)
    x = Conv2D(64, (3, 3), padding='valid', name='block1_conv8')(x)
    x = BatchNormalization()(x)

    x = Conv2D(CATEGORIES, kernel_size=(1,1), padding='valid', name='block1_conv9')(x)
    x = Reshape((CATEGORIES, IMAGE_DATA_SHAPE[0]*IMAGE_DATA_SHAPE[1]), input_shape=(CATEGORIES, IMAGE_DATA_SHAPE[0], IMAGE_DATA_SHAPE[1]))(x)
    x = Permute((2,1))(x)
    x = Activation('softmax')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='segnet_basic_functional')

    with open(MODEL_PATH+'basic_model.json', 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

    return model

def create_model(input_shape=None):
    """Instantiates the SegNet architecture using the Model API.
    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels, and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid input shape.
    """
    print('Creating Model using Model API')
    input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=48, data_format=K.image_data_format())
    
    model = Sequential()
    # encoding layers
    model.add(Convolution2D(FILTER_SIZE, kernel_size= KERNEL_SIZE, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE, padding='valid'))

    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(2*FILTER_SIZE, kernel_size=KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(4*FILTER_SIZE, kernel_size=KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(8*FILTER_SIZE, kernel_size=KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # decoding layers
    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(8*FILTER_SIZE, kernel_size=KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=POOL_SIZE))
    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(4*FILTER_SIZE, kernel_size=KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=POOL_SIZE))
    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(2*FILTER_SIZE, kernel_size=KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=POOL_SIZE))
    model.add(ZeroPadding2D(padding=PAD_SIZE))
    model.add(Convolution2D(FILTER_SIZE, kernel_size=KERNEL_SIZE, padding='valid'))
    model.add(BatchNormalization())

    model.add(Convolution2D(CATEGORIES, kernel_size=(1,1), padding='valid'))
    # model.add(Reshape((12,360*480), input_shape=(12,360,480)))
    model.add(Reshape((CATEGORIES, IMAGE_DATA_SHAPE[0]*IMAGE_DATA_SHAPE[1]), input_shape=(CATEGORIES, IMAGE_DATA_SHAPE[0], IMAGE_DATA_SHAPE[1])))
    model.add(Permute((2,1))) #, input_shape=(12,360*480)))
    model.add(Activation('softmax'))

    with open(MODEL_PATH+'basic_model.json', 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

    return model

def train_model(segnet_basic_model):
    # load model architecture
    # with open(MODEL_PATH+'basic_model.json', 'r') as model_file:
    #     segnet_basic_model = model_from_json(model_file.read())
    # load training and test data
    with h5py.File(DATA_PATH+DATASET_FILE, 'r') as hf:
        train_data = hf['train_data'][:]
        train_label = hf['train_label'][:]
        test_data = hf['test_data'][:]
        test_label = hf['test_label'][:]
        val_data = hf['val_data'][:]
        val_label = hf['val_label'][:]

    train_dataT = train_data.transpose(0,2,3,1)
    val_dataT = val_data.transpose(0,2,3,1)    
    test_dataT = test_data.transpose(0,2,3,1)

    # defining checkpoints
    model_checkpoint = ModelCheckpoint(MODEL_PATH+'basic_model_{epoch:03d}.hdf5')
    csv_log = CSVLogger(MODEL_PATH+'basic_model_training_log.csv', separator=',', append=False)
    callbacks = [model_checkpoint, csv_log]

    print('Compling Model')
    segnet_basic_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Training Model')
    history = segnet_basic_model.fit(train_dataT, 
                                    train_label, 
                                    callbacks=callbacks, 
                                    batch_size=BATCH_SIZE, 
                                    epochs=NUM_EPOCH, 
                                    verbose=1,
                                    class_weight= CLASS_WEIGHTING,
                                    validation_data=(val_dataT, val_label), #val split=1/3
                                    shuffle=True)


if __name__ == '__main__':
    parse_and_set_arguments()
    if MODEL_TYPE:
        segnet_basic_model = create_model(input_shape=IMAGE_DATA_SHAPE)
    else:
        segnet_basic_model = create_model_func(input_shape=IMAGE_DATA_SHAPE)
    # segnet_basic_model.summary()
    train_model(segnet_basic_model)





