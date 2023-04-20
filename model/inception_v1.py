from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def stage_4(x):
    params = [[192, 96, 208, 16, 48, 64],
              [160, 112, 224, 24, 64, 64],
              [128, 128, 256, 24, 64, 64],
              [112, 144, 288, 32, 64, 64],
              [256, 160, 320, 32, 128, 128]]

    for i in range(len(params)):
        if i == len(params)-1:
            x = inception_module(x, params[i])
        else:
            x = inception_module(x, params[i])
            x = inception_module(x, params[i])

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x


def inception_v1(input_shape, num_classes):
    input_img = Input(shape=input_shape)

    x = stage_1(input_img)
    x = stage_2(x)
    x = stage_3(x)
    x = stage_4(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=output)
    return model


def train():
    train_dir = 'data/train'
    valid_dir = 'data/valid'
    test_dir = 'data/test'

    target_size = (224, 224)
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(train_dir,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

    valid_data = valid_datagen.flow_from_directory(valid_dir,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

    test_data = test_datagen.flow_from_directory(test_dir,
                                                 target_size=target_size,
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

    train_data, valid_data, test_data
    model = inception_v1(input_shape=(224, 224, 3), num_classes=5)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    epochs = 10
    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=valid_data)
