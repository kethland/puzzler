import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, concatenate
from data_loader import load_data

import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from googletrans import Translator


# Load the Inception V3.5 model
model = load_model('inception_v3.5.h5')


def load_image_from_url(url):
    """
    Loads an image from a URL and returns it as a NumPy array.
    Args:
        url: The URL of the image to load.
    Returns:
        A NumPy array of the loaded image.
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(url):
    """
    Predicts the class of an image from a URL using the Inception V3.5 model.
    Args:
        url: The URL of the image to predict.
    Returns:
        The predicted class label and probability.
    """
    img_array = load_image_from_url(url)
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]
    label = np.argmax(pred)
    prob = pred[label]
    return label, prob


def translate_message(message):
    """
    Translates a message from any language to English using Google Translate.
    Args:
        message: The message to translate.
    Returns:
        The translated message in English.
    """
    translator = Translator(service_urls=['translate.google.com'])
    result = translator.translate(message, dest='en')
    return result.text


def decrypt_message(message):
    gematria_table = {
        'ᚠ': 1, 'ᚢ': 2, 'ᚦ': 3, 'ᚨ': 4, 'ᚱ': 5, 'ᚲ': 6, 'ᚷ': 7, 'ᚹ': 8, 'ᚺ': 9,
        'ᚾ': 10, 'ᛁ': 11, 'ᛃ': 12, 'ᛇ': 13, 'ᛈ': 14, 'ᛉ': 15, 'ᛊ': 16, 'ᛏ': 17, 'ᛒ': 18,
        'ᛖ': 19, 'ᛗ': 20, 'ᛚ': 21, 'ᛝ': 22, 'ᛞ': 23, 'ᛟ': 24
    }
    decrypted_message = ""
    for char in message:
        if char in gematria_table:
            decrypted_message += chr(gematria_table[char] + 64)
    return decrypted_message


def process_image(url):
    """
    Processes an image from a URL and returns the decrypted and translated message.
    Args:
        url: The URL of the image to process.
    Returns:
        The decrypted and translated message.
    """
    label, prob = predict_image(url)
    if label == 0:
        message = "This is not a valid glyph."
    else:
        message = "The glyph represents the letter " + chr(label + 64)
    decrypted_message = decrypt_message(message)
    translated_message = translate_message(decrypted_message)
    return translated_message


if __name__ == '__main__':
    url = input("Enter the URL of the image to process: ")
    message = process_image(url)
    print(message)


