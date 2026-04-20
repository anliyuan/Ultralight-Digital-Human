def square_crop_from_landmarks(landmarks, image_shape):
    xmin = int(landmarks[1][0])
    ymin = int(landmarks[52][1])
    xmax = int(landmarks[31][0])
    width = xmax - xmin
    ymax = ymin + width

    if width <= 0:
        raise ValueError("invalid crop width from landmarks")

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
