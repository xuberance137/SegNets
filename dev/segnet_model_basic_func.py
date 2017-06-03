# from __future__ import absolute_import
# from __future__ import print_function

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

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.utils.data_utils import get_file

from keras.models import Sequential, model_from_json
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import SGD
from keras import backend as K
import json
import cv2
import numpy as np

np.random.seed(007) # starting seed for reproducibility

def parse_and_set_arguments():
    global TRAINING
    parser = argparse.ArgumentParser(description='Basic Segnet Model Training and Test App')
    parser.add_argument('--train', type=int, action='store', dest='TRAINING', default=1, help='Run Segnet in train or test mode(default is test)')
    args = parser.parse_args()
    TRAINING = args.TRAINING

def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        include_top):
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
    if include_top:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True`, '
                                 '`input_shape` should be ' + str(default_shape) + '.')
        input_shape = default_shape
    else:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3:
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + ', got '
                                     '`input_shape=' + str(input_shape) + '`')
            else:
                input_shape = (3, None, None)
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3:
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + ', got '
                                     '`input_shape=' + str(input_shape) + '`')
            else:
                input_shape = (None, None, 3)
    return input_shape

def create_model(include_top=False, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
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

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='segnet_basic_functional')

    # load weights
    if weights == 'imagenet':
        # if include_top:
        #     weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
        #                             WEIGHTS_PATH,
        #                             cache_subdir='models')
        # else:
        #     weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #                             WEIGHTS_PATH_NO_TOP,
        #                             cache_subdir='models')
        if include_top:
            weights_path = '../model/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        else:
            weights_path = '../model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'   

        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
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
                                    class_weight= CLASS_WEIGHTING,
                                    validation_data=(val_data, val_label), #val split=1/3
                                    shuffle=True)


if __name__ == '__main__':
    parse_and_set_arguments()
    #create_model()
    segnet_basic_model = create_model(input_shape=IMAGE_DATA_SHAPE, weights=None)
    segnet_basic_model.summary()
    #train_model(segnet_basic_model)

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





