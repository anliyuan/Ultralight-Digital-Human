import cv2


DEFAULT_MASK_LEFT_RATIO = 5 / 160
DEFAULT_MASK_TOP_RATIO = 5 / 160
DEFAULT_MASK_RIGHT_RATIO = 150 / 160
DEFAULT_MASK_BOTTOM_RATIO = 145 / 160


def apply_blackout_mask(
    image,
    left_ratio=DEFAULT_MASK_LEFT_RATIO,
    top_ratio=DEFAULT_MASK_TOP_RATIO,
    right_ratio=DEFAULT_MASK_RIGHT_RATIO,
    bottom_ratio=DEFAULT_MASK_BOTTOM_RATIO,
):
    if not (0 <= left_ratio < right_ratio <= 1):
        raise ValueError("mask x ratios must satisfy 0 <= left < right <= 1")
    if not (0 <= top_ratio < bottom_ratio <= 1):
        raise ValueError("mask y ratios must satisfy 0 <= top < bottom <= 1")

    height, width = image.shape[:2]
    x1 = int(round(width * left_ratio))
    y1 = int(round(height * top_ratio))
    x2 = int(round(width * right_ratio))
    y2 = int(round(height * bottom_ratio))
    return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0, 0, 0), -1)
