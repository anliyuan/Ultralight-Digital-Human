def expanded_square_crop_from_landmarks(landmarks, image_shape, expand_ratio=1.0):
    if expand_ratio <= 0:
        raise ValueError("expand_ratio must be positive.")

    xmin = float(landmarks[1][0])
    ymin = float(landmarks[52][1])
    xmax = float(landmarks[31][0])
    width = xmax - xmin
    ymax = ymin + width

    if width <= 0:
        raise ValueError("invalid crop width from landmarks")

    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    side = width * expand_ratio

    xmin = int(round(center_x - side / 2.0))
    xmax = int(round(center_x + side / 2.0))
    ymin = int(round(center_y - side / 2.0))
    ymax = int(round(center_y + side / 2.0))

    height, img_width = image_shape[:2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(height, ymax)

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("crop box collapsed after clamping to image bounds")

    side = min(xmax - xmin, ymax - ymin)
    xmax = xmin + side
    ymax = ymin + side
    return xmin, ymin, xmax, ymax
