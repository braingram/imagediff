import numpy

import loader


def distance(image1, image2):
    image1 = loader.load(image1)
    image2 = loader.load(image2)
    return numpy.sqrt(numpy.sum((image1 - image2) ** 2.))

