import globals
from keras import optimizers
import keras.layers as layers
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Input, BatchNormalization, Activation, Concatenate, Conv2DTranspose
from keras.models import Model


def block(block_type, input_tensor, kernel_size, filters, stage, block_name, strides=(2, 2)):
    """ Creates an Identity or Conv layer for the ResNet50 model.

    The identity block is the block that has no conv layer at shortcut.
    The conv block is the block that has a conv layer at shortcut.

    # Arguments
        block_type: convolution or identity
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block_name: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    # References
        ResNet50 Keras implementation
        https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
    """

    filters1, filters2, filters3 = filters
    x = 0
    conv_name_base = 'res' + str(stage) + block_name + '_branch'
    bn_name_base = 'bn' + str(stage) + block_name + '_branch'

    if block_type is globals.IDENTITY:
        x = Convolution2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    elif block_type is globals.CONV:
        x = Convolution2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
        x = UpSampling2D(strides, name=conv_name_base + '2a_sample')(x)

    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    if block_type is globals.CONV:
        shortcut = Convolution2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
        shortcut = UpSampling2D(strides, name=conv_name_base + '1_sample')(shortcut)
        shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)
        x = layers.add([x, shortcut])
    if block_type is globals.IDENTITY:
        x = layers.add([x, input_tensor])

    x = Activation('relu')(x)
    return x


def fire_block(x, filters, name="fire", orientation='encoder'):
    """ Creates a fire block for the SqueezeNet model.

        # Arguments
            x: input tensor
            filters: list of integers, the filters of 3 conv layer at main path
        # Returns
            Output tensor for the block.
        # References
            SqueezeNet implementation
            https://github.com/wohlert/keras-squeezenet/blob/master/squeezenet.py
    """

    sq_filters, ex1_filters, ex2_filters = filters
    if orientation is 'encoder':
        # squeeze layer
        squeeze = Convolution2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
        # both expand layers
        expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
        expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
        # concatenate both and return the result
        x = Concatenate(axis=-1, name=name+'/Merge')([expand1, expand2])
    else:
        expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(x)
        expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(x)
        squeeze = Concatenate(axis=-1, name=name+'/Merge')([expand1, expand2])
        x = Convolution2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(squeeze)
    return x


def create_compile_model(input_shape, model_type, model_optimizer=optimizers.SGD(lr=0.01, nesterov=True)):
    """ Creates and compiles the defined model.

        Creates the model depending on the type as parameter, then compiles it
        with given parameters.

        # Arguments
            input_shape: input shape
            model_type: VGG/RESNET/SQUEEZENET
            model_optimizer: optimizer for the model compilation
            filters: list of integers, the filters of 3 conv layer at main path
        # Returns
            Compiled defined model
        # References
            VGG16 implementation
            https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
    """

    print('\nCompiling model...')
    # create the model and compile it
    inputs = Input(shape=input_shape)
    x = 0

    if model_type is 'simple':
        x = Convolution2D(16, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
        x = Convolution2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = Convolution2D(16, (3, 3), activation='relu', padding='same', name='conv3')(x)
        encoded = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool')(x)

        x = Convolution2D(16, (3, 3), activation='relu', padding='same', name='deconv1')(encoded)
        x = Convolution2D(16, (3, 3), activation='relu', padding='same', name='deconv2')(x)
        x = Convolution2D(16, (3, 3), activation='relu', padding='same', name='deconv3')(x)
        x = UpSampling2D((2, 2), name='sample')(x)

    if model_type is globals.VGG:
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        encoded = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_deconv1')(encoded)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_deconv2')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_deconv3')(x)
        x = UpSampling2D((2, 2), name='block5_sample')(x)

        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_deconv1')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_deconv2')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_deconv3')(x)
        x = UpSampling2D((2, 2), name='block4_sample')(x)

        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_deconv1')(x)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_deconv2')(x)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_deconv3')(x)
        x = UpSampling2D((2, 2), name='block3_sample')(x)

        x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_deconv1')(x)
        x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_deconv2')(x)
        x = UpSampling2D((2, 2), name='block2_sample')(x)

        x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_deconv1')(x)
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_deconv2')(x)
        x = UpSampling2D((2, 2), name='block1_sample')(x)

    if model_type is globals.RESNET:
        x = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2), name='conv1_sample1')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='conv1_pool')(x)
        x = UpSampling2D((2, 2), name='conv1_sample2')(x)
        x = ZeroPadding2D((1, 1), name='conv1_padding')(x)

        x = block(globals.CONV, x, 3, (64, 64, 256), stage=2, block_name='a', strides=(1, 1))
        x = block(globals.IDENTITY, x, 3, (64, 64, 256), stage=2, block_name='b')
        x = block(globals.IDENTITY, x, 3, (64, 64, 256), stage=2, block_name='c')

        x = block(globals.CONV, x, 3, (128, 128, 512), stage=3, block_name='a')
        x = block(globals.IDENTITY, x, 3, (128, 128, 512), stage=3, block_name='b')
        x = block(globals.IDENTITY, x, 3, (128, 128, 512), stage=3, block_name='c')
        x = block(globals.IDENTITY, x, 3, (128, 128, 512), stage=3, block_name='d')

        x = block(globals.CONV, x, 3, (256, 256, 1024), stage=4, block_name='a')
        x = block(globals.IDENTITY, x, 3, (256, 256, 1024), stage=4, block_name='b')
        x = block(globals.IDENTITY, x, 3, (256, 256, 1024), stage=4, block_name='c')
        x = block(globals.IDENTITY, x, 3, (256, 256, 1024), stage=4, block_name='d')
        x = block(globals.IDENTITY, x, 3, (256, 256, 1024), stage=4, block_name='e')
        x = block(globals.IDENTITY, x, 3, (256, 256, 1024), stage=4, block_name='f')

        x = block(globals.CONV, x, 3, (512, 512, 2048), stage=5, block_name='a')
        x = block(globals.IDENTITY, x, 3, (512, 512, 2048), stage=5, block_name='b')
        x = block(globals.IDENTITY, x, 3, (512, 512, 2048), stage=5, block_name='c')

    if model_type is globals.SQUEEZENET:
        x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', name='conv1')(inputs)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_block(x, (16, 64, 64), name='fire2')
        x = fire_block(x, (16, 64, 64), name='fire3')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_block(x, (32, 128, 128), name='fire4')
        x = fire_block(x, (32, 128, 128), name='fire5')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_block(x, (48, 192, 192), name='fire6')
        x = fire_block(x, (48, 192, 192), name='fire7')

        x = fire_block(x, (64, 256, 256), name='fire8')
        encoded = fire_block(x, (64, 256, 256), name='fire9')

        x = fire_block(encoded, (64, 256, 256), name='opp-fire9', orientation='decoder')
        x = fire_block(x, (64, 256, 256), name='opp-fire8', orientation='decoder')

        x = fire_block(x, (48, 192, 192), name='opp-fire7', orientation='decoder')
        x = fire_block(x, (48, 192, 192), name='opp-fire6', orientation='decoder')

        x = UpSampling2D((2, 2), name='sample5')(x)
        x = ZeroPadding2D((1, 1), name='padding5')(x)

        x = fire_block(x, (32, 128, 128), name='opp-fire5', orientation='decoder')
        x = fire_block(x, (32, 128, 128), name='opp-fire4', orientation='decoder')
        x = UpSampling2D((2, 2), name='sample3')(x)

        x = fire_block(x, (16, 64, 64), name='opp-fire3', orientation='decoder')
        x = fire_block(x, (16, 64, 64), name='opp-fire2', orientation='decoder')
        x = UpSampling2D((2, 2), name='sample2')(x)

        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', name='conv-1')(x)

    # last convolution to have last layer output an image
    decoded = Convolution2D(1, (3, 3), activation='sigmoid', padding='same', name='final_conv')(x)

    model_created = Model(inputs, decoded, name=model_type)

    # compile the model with given loss and optimizer
    model_created.compile(loss='binary_crossentropy',
                          optimizer=model_optimizer,
                          metrics=['accuracy'])
    return model_created

