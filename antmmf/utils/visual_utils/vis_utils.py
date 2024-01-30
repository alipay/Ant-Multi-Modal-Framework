# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import torch


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def visualize(
    image,
    coords,
    classes=None,
    class_names=None,
    confs=None,
    label_format="xywh",
    normalized=True,
    image_format="RGB",
    save_path=None,
):
    """
    Args:
        image(torch.Tensor or np.ndArray): image with shape [h,w,c]
        coords(torch.Tensor or np.ndArray): Nx4, [xc,yc,w,h] or [x1,y1,x2,y2]
        label_format(str): xywh or x1y1x2y2
        normalized(bool): whether input label coords are already normalized

    Returns:

    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    image = image.copy()
    coords = coords.copy()
    if image_format == "RGB":
        image = image[:, :, ::-1]
    height, width = image.shape[:2]
    if normalized:
        coords[:, [0, 2]] *= width
        coords[:, [1, 3]] *= height
    if label_format == "xywh":
        vis_coords = coords.copy()
        vis_coords[:, 0] = coords[:, 0] - coords[:, 2] / 2.0
        vis_coords[:, 2] = coords[:, 0] + coords[:, 2] / 2.0
        vis_coords[:, 1] = coords[:, 1] - coords[:, 3] / 2.0
        vis_coords[:, 3] = coords[:, 1] + coords[:, 3] / 2.0
    elif label_format == "x1y1x2y2":
        vis_coords = coords
    else:
        raise Exception("unknown format")

    colors_num = vis_coords.shape[0]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(colors_num)]

    for i, xyxy in enumerate(vis_coords):
        color = colors[i]
        label = ""
        if classes is not None:
            # int or str
            label = classes[i] if not class_names else class_names[classes[i]]
            label = str(label)
        if confs is not None:
            conf = confs[i]  # float
            label += " %.2f" % conf
        plot_one_box(xyxy, image, label=label, color=color, line_thickness=3)
    if save_path:
        print(f"saving to:{save_path}")
        cv2.imwrite(save_path, image)
    return image
