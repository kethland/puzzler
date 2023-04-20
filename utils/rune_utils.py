def decrypt_message(message):
    """
    Decrypts a message encoded using the Runic script.
    Args:
        message: The message to decrypt.
    Returns:
        The decrypted message.
    """
    rune_to_eng = {
        'ᚠ': 'f', 'ᚢ': 'u', 'ᚦ': 'th', 'ᚨ': 'a', 'ᚱ': 'r', 'ᚲ': 'k', 'ᚷ': 'g', 'ᚹ': 'w',
        'ᚺ': 'h', 'ᚾ': 'n', 'ᛁ': 'i', 'ᛃ': 'j', 'ᛇ': 'eo', 'ᛈ': 'p', 'ᛉ': 'x', 'ᛊ': 's',
        'ᛏ': 't', 'ᛒ': 'b', 'ᛖ': 'e', 'ᛗ': 'm', 'ᛚ': 'l', 'ᛝ': 'ng', 'ᛞ': 'd', 'ᛟ': 'oe'
    }
    decrypted_message = ""
    for char in message:
        if char in rune_to_eng:
            decrypted_message += rune_to_eng[char]
        else:
            decrypted_message += char
    return decrypted_message
