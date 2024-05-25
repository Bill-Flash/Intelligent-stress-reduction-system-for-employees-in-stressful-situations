from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import PReLU


def CNN1(input_shape=(48, 48, 1), n_classes=8):
    # input
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    # block2
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    # block3
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    # fc
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def CNN2(input_shape=(48, 48, 1), n_classes=8):
    """
    Referencing the paper 'Going Deeper with Convolutions', add a 1x1 convolution layer after the
    input layer to increase non-linear representation

    :param input_shape:
    :param n_classes:
    :return:
    """
    # input
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(32, (5, 5), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block3
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


def CNN3(input_shape=(48, 48, 1), n_classes=8):
    """
    Reference paper:
    A Compact Deep Learning Model for Robust Facial Expression Recognition
    :param input_shape:
    :param n_classes:
    :return:
    """
    # input
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    # block1
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model
