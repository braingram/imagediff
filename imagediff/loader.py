import os

import PIL.Image

import numpy

def load(image):
    # if it's already an array, do nothing
    if type(image) == numpy.ndarray: return image
    # if it's a list, make it an array
    if type(image) == list: return numpy.array(image)
    # if it's not an array, load it
    if type(image) == str: return load_from_string(image)
    raise TypeError("Unknown type: %s" % str(type(image)))

def load_from_string(image):
    fn, ext = os.path.splitext(image)
    ext = ext.lower()
    # handle 'special' filetypes here
    #if ext == '.jpg':
    #    return load_image_file_jpg(image)
    return load_image_file(image)

def pil_to_array(image):
    if image.mode == 'LA':
        # pil does not like LA
        shape, typestr = ((image.size[1], image.size[0], 2), '|u1')
        return numpy.fromstring(image.tostring(), '|u1').reshape(\
                (image.size[1], image.size[0], 2))
    else:
        return numpy.array(image)

def load_image_file(image):
    return pil_to_array(PIL.Image.open(image))
