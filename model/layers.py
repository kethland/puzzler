from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, concatenate, GlobalAveragePooling2D

def conv2d_bn(x, filters, kernel_size, padding='same', strides=(1, 1), activation='relu', name=None):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activation, name=name)(x)
    x = BatchNormalization()(x)
    return x


def inception_module(x, params):
    tower_1 = conv2d_bn(x, params[0], (1, 1))

    tower_2 = conv2d_bn(x, params[1], (1, 1))
    tower_2 = conv2d_bn(tower_2, params[2], (3, 3))

    tower_3 = conv2d_bn(x, params[3], (1, 1))
    tower_3 = conv2d_bn(tower_3, params[4], (5, 5))

    tower_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_4 = conv2d_bn(tower_4, params[5], (1, 1))

    output = concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
    return output


def stage_1(input_img):
    x = conv2d_bn(input_img, 64, (7, 7), strides=(2, 2))
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x  


def stage_2(x):
    params = [[64, 96, 128, 16, 32, 32],
              [128, 128, 192, 32, 96, 64]]

    for p in params:
        x = inception_module(x, p)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x

   

def stage_3(x):
    params = [[192, 96, 208, 16, 48, 64],
              [160, 112, 224, 24, 64, 64],
              [128, 128, 256, 24, 64, 64]]

    for p in params:
        x = inception_module(x, p)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x
