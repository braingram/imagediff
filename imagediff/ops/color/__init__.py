import numpy

def to_gray(image, wts=(.299, .587, .114)):
    # from wikipedia
    # 30% of the red value, 59% of the green value, and 11%
    # from image magick
    # 0.29900*R+0.58700*G+0.11400*B
    dtype = image.dtype
    return numpy.sum(image * wts,2).astype(dtype)
