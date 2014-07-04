import numpy
from ..loader import load


def apply_background(image, bg=0.5):
    image = load(image)
    bg = make_background_from_image(bg, image)
    if image.ndim == 2:
        raise ValueError("image on has 1 channel, unknown alpha")
    if image.shape[2] == 4:  # RGBA
        return rgba_apply_background(image, bg)
    elif image.shape[2] == 2:  # LA
        return la_apply_background(image, bg)
    else:
        raise ValueError("image depth != [4 or 2], unknown alpha: %i" %
                         image.shape[2])


def make_background_from_image(bg, image):
    if type(bg) in [float, int]:  # singular
        return (numpy.ones(image.shape[:2]) * bg).astype(image.dtype)
    elif type(bg) in [tuple, list, numpy.ndarray]:
        if len(bg) == 3:  # RGB
            return (numpy.ones((image.shape[0], image.shape[1], 3))
                    * bg).astype(image.dtype)
    bg = load(bg)
    if bg.shape[:2] != image.shape[:2]:
        raise ValueError("background was incorrect shape: %s" %
                         str(bg.shape))
    return bg


def rgba_apply_background(image, bg):
    bwt = 1 - image[:, :, 3]/255.
    bwt = numpy.dstack((bwt, bwt, bwt))
    iwt = 1 - bwt
    return (image[:, :, :3] * iwt + bg * bwt).astype(image.dtype)


def la_apply_background(image, bg):
    bwt = 1 - image[:, :, 1]/255.
    # bwt = numpy.dstack((bwt, bwt, bwt))
    iwt = 1 - bwt
    return (image[:, :, 0] * iwt + bg * bwt).astype(image.dtype)
