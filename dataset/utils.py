import numpy as np
from PIL import Image


def inv_depths(depth_min, depth_max, n_depths):
    depths = 1. / np.linspace(1 / depth_max, 1 / depth_min, n_depths).astype(np.float32)
    return depths


def scale_img_cam(img_HW3, cam_244, new_h, new_w):
    old_w, old_h = img_HW3.size
    img_HW3 = img_HW3.resize((new_w, new_h), Image.BILINEAR)
    img_hw3 = np.array(img_HW3, dtype=np.float32)

    scale_h = new_h / old_h
    scale_w = new_w / old_w

    new_cam_244 = cam_244.copy()
    # focal
    new_cam_244[0, 0, 0] *= (scale_w * 0.5)
    new_cam_244[0, 1, 1] *= (scale_h * 0.5)
    # principal point
    new_cam_244[0, 0, 2] *= (scale_w * 0.5)
    new_cam_244[0, 1, 2] *= (scale_h * 0.5)

    return img_hw3, new_cam_244