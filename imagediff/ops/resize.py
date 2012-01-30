import numpy

from ..loader import load

def match_sizes(image1, image2, fill):
    image1 = load(image1)
    image2 = load(image2)

    size1 = image1.shape[:2]
    size2 = image2.shape[:2]

    newsize = [max(size1[i],size2[i]) for i in xrange(2)]

    return resize(image1, newsize, fill),\
            resize(image2, newsize, fill)

def calculate_border(image, size):
    id0 = image.shape[0]
    s0 = size[0]
    id1 = image.shape[1]
    s1 = size[1]
    b0, r0 = divmod(s0 - id0, 2)
    b1, r1 = divmod(s1 - id1, 2)
    return (b0, b0 + r0, b1, b1 + r1)

def resize(image, size, fill):
    if (image.shape[0] > size[0]) or (image.shape[1] > size[1]):
        raise ValueError("Image shape(%s) was larger than size(%s)" % \
                (str(image.shape), str(size)))
    top, bottom, left, right = calculate_border(image, size)
    filler = (numpy.ones_like(image) * fill).astype(image.dtype)

    # add top & bottom border
    image = numpy.vstack((filler[:top,:], image, filler[:bottom,:]))

    filler = (numpy.ones_like(image) * fill).astype(image.dtype)

    # add left & right border
    image = numpy.hstack((filler[:,:left], image, filler[:,:right]))

    return image
