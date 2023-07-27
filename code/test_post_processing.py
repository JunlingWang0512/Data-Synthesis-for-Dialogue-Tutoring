# import string

# def remove_unexpected_chars(text):
#     keep = string.printable + 'Â°'
#     return ''.join(char for char in text if char in keep)

# text = 'â€79Â°270'

# text = remove_unexpected_chars(text)
# print("After:", text)
def remove_unexpected_chars(text):
    # specify the list of characters to remove
    chars_to_remove = 'â€€â€œ'
    # create a translation table that maps every character in chars_to_remove to None
    trans = str.maketrans('', '', chars_to_remove)
    # apply the translation table to the text
    return text.translate(trans)

text = 'â€79Â°270'

text = remove_unexpected_chars(text)
print("After:", text)
