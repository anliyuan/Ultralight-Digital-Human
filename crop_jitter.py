import random


def jitter_crop_box(xmin, ymin, xmax, ymax, image_shape, jitter_ratio=0.0, rng=None):
    if jitter_ratio < 0:
        raise ValueError("jitter_ratio must be non-negative.")
    if rng is None:
        rng = random

    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError("crop box must have positive size.")

    max_jitter_x = int(round(width * jitter_ratio))
    max_jitter_y = int(round(height * jitter_ratio))
    shift_x = rng.randint(-max_jitter_x, max_jitter_x)
    shift_y = rng.randint(-max_jitter_y, max_jitter_y)

    xmin += shift_x
    xmax += shift_x
    ymin += shift_y
    ymax += shift_y

    img_h, img_w = image_shape[:2]
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    if ymin < 0:
        ymax -= ymin
        ymin = 0
    if xmax > img_w:
        xmin -= xmax - img_w
        xmax = img_w
    if ymax > img_h:
        ymin -= ymax - img_h
        ymax = img_h

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)
    return xmin, ymin, xmax, ymax
