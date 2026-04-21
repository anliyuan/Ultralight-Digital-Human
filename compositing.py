import cv2
import numpy as np


def feather_alpha(shape, feather=12):
    height, width = shape[:2]
    mask = np.ones((height, width), dtype=np.float32)
    if feather <= 0:
        return mask[:, :, None]

    mask[:feather, :] *= np.linspace(0.0, 1.0, feather, dtype=np.float32)[:, None]
    mask[-feather:, :] *= np.linspace(1.0, 0.0, feather, dtype=np.float32)[:, None]
    mask[:, :feather] *= np.linspace(0.0, 1.0, feather, dtype=np.float32)[None, :]
    mask[:, -feather:] *= np.linspace(1.0, 0.0, feather, dtype=np.float32)[None, :]
    return mask[:, :, None]


def blend_patch(base_patch, generated_patch, feather=12):
    if base_patch.shape != generated_patch.shape:
        raise ValueError("base_patch and generated_patch must have the same shape")
    alpha = feather_alpha(base_patch.shape, feather=feather)
    blended = generated_patch.astype(np.float32) * alpha + base_patch.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def hard_crop_composite(frame, crop_img_ori, pred, bbox, inner_margin=4):
    xmin, ymin, xmax, ymax = bbox
    crop_h = ymax - ymin
    crop_w = xmax - xmin
    crop_img_ori[inner_margin:-inner_margin, inner_margin:-inner_margin] = pred
    pasted = cv2.resize(crop_img_ori, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    frame[ymin:ymax, xmin:xmax] = pasted
    return frame


def feathered_crop_composite(frame, crop_img_ori, pred, bbox, inner_margin=4, feather=12):
    xmin, ymin, xmax, ymax = bbox
    crop_h = ymax - ymin
    crop_w = xmax - xmin
    base_inner = crop_img_ori[inner_margin:-inner_margin, inner_margin:-inner_margin].copy()
    crop_img_ori[inner_margin:-inner_margin, inner_margin:-inner_margin] = blend_patch(
        base_inner, pred, feather=feather
    )
    pasted = cv2.resize(crop_img_ori, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    frame[ymin:ymax, xmin:xmax] = pasted
    return frame


def composite_prediction(
    frame,
    crop_img_ori,
    pred,
    bbox,
    mode="hard_crop",
    inner_margin=4,
    feather=12,
):
    if mode == "hard_crop":
        return hard_crop_composite(frame, crop_img_ori, pred, bbox, inner_margin=inner_margin)
    if mode == "feathered_crop":
        return feathered_crop_composite(
            frame, crop_img_ori, pred, bbox, inner_margin=inner_margin, feather=feather
        )
    raise ValueError(f"unsupported composite mode: {mode}")
