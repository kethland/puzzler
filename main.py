import numpy as np
import requests
from PIL import Image
from io import BytesIO
from googletrans import Translator

def load_image(url):
    img_response = requests.get(url)
    img = Image.open(BytesIO(img_response.content))
    img_array = np.array(img)
    return img_array

def translate_text(text):
    translator = Translator(service_urls=['translate.google.com'])
    result = translator.translate(text, dest='en')
    return result.text

def decode_message(message):
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

import numpy as np

def modulus_decrypt(ciphertext, modulus):
    decrypted_blocks = []
    for i, char in enumerate(ciphertext):
        if char.isalpha():
            decrypted_char = chr((ord(char) - 65) % modulus[i] + 65)
        else:
            decrypted_char = char
        decrypted_blocks.append(ord(decrypted_char) - 65)
    
    # Chinese Remainder Theorem
    M = np.prod(modulus)
    decrypted_message = 0
    for i, b in enumerate(decrypted_blocks):
        mi = M // modulus[i]
        mi_inv = pow(mi, -1, modulus[i])
        decrypted_message += b * mi * mi_inv
    
    return ''.join([chr(decrypted_message % 26 + 65)])


def substitution_decrypt(ciphertext, substitution_key):
    decrypted_message = ""
    for char in ciphertext:
        if char.isalpha():
            if char.lower() in substitution_key:
                decrypted_char = substitution_key[char.lower()]
                if char.isupper():
                    decrypted_char = decrypted_char.upper()
            else:
                decrypted_char = char
        else:
            decrypted_char = char
        decrypted_message += decrypted_char
    return decrypted_message

def process_image(image_url, modulus=None, substitution_key=None):
    img_array = load_image(image_url)
    message = "The glyph represents the letter " + chr(img_array[0][0][0] + 64)
    decoded_message = decode_message(message)
    translated_message = translate_text(decoded_message)
    if modulus is not None:
        decrypted_message = modulus_decrypt(translated_message, modulus)
        translated_message = decrypted_message
    if substitution_key is not None:
        decrypted_message = substitution_decrypt(translated_message, substitution_key)
        translated_message = decrypted_message
    return translated_message
