import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_from_directory(data_dirs, batch_size=32, image_size=(224, 224)):
    """
    Loads image data from multiple directories using ImageDataGenerator.
    Args:
        data_dirs: List of directory paths where the data is located.
        batch_size: Number of images to load in each batch (default=32).
        image_size: Tuple containing the size of the images (default=(224, 224)).
    Returns:
        A tuple containing the loaded image data and the number of classes.
    """
    # Create a list of sub-directories in each data directory
    sub_dirs = []
    for data_dir in data_dirs:
        sub_dirs.append([os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    # Combine all sub-directories into a single list
    sub_dirs = [sub_dir for dirs in sub_dirs for sub_dir in dirs]

    # Use ImageDataGenerator to load the images
    datagen = ImageDataGenerator(rescale=1./255)
    data = datagen.flow_from_directory(directory=sub_dirs[0],
                                        target_size=image_size,
                                        class_mode='categorical',
                                        batch_size=batch_size)
    for sub_dir in sub_dirs[1:]:
        data_dir = datagen.flow_from_directory(directory=sub_dir,
                                                target_size=image_size,
                                                class_mode='categorical',
                                                batch_size=batch_size)
        data = data.concatenate(data_dir)

    # Get the number of classes
    num_classes = len(data.class_indices)

    return data, num_classes

data_dirs = ['path/to/dataset1', 'path/to/dataset2', 'path/to/dataset3', 'path/to/dataset4']
batch_size = 32
image_size = (224, 224)
train_data, num_classes = load_data_from_directory(data_dirs, batch_size=batch_size, image_size=image_size)

