# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from .palette import get_palette

dirname = os.path.dirname(__file__)
Font = ImageFont.truetype(os.path.join(dirname, "huawenfangsong.ttf"), 20)


def order_points(pts):
    pts = np.array(pts)
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    tl, bl = leftMost[np.argsort(leftMost[:, 1]), :]
    tr, br = rightMost[np.argsort(rightMost[:, 1]), :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    """
    interface that supports chinese text visualization
    """
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text((left, top), text, textColor, font=Font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_multiple_line_text(image, text, text_color, text_start_height):
    """
    refer to:
    python PIL draw multiline text on image:
    https://stackoverflow.com/a/7698300/395857
    """
    import textwrap

    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    y_text = text_start_height
    lines = textwrap.wrap(text, width=40)
    for line in lines:
        line_width, line_height = Font.getsize(line)
        draw.text(
            ((image_width - line_width) / 2, y_text), line, font=Font, fill=text_color
        )
        y_text += line_height
    return y_text


class VisTool(object):
    def __init__(self, image_np_or_path):
        if isinstance(image_np_or_path, np.ndarray):
            self._canvas = image_np_or_path
        else:
            self._canvas = cv2.imread(image_np_or_path)

    def save(self, savepath):
        if len(savepath.split("/")) >= 2:
            savedir = osp.dirname(savepath)
            if not osp.exists(savedir):
                os.makedirs(savedir)
        cv2.imwrite(savepath, self._canvas)

    @property
    def canvas(self):
        return self._canvas

    def draw(self, coords, color=[255, 0, 0], prefix=None, canvas=None):

        if coords.shape[0] == 0:
            return self._canvas
        if coords.shape[1] == 8:
            coord = coords[:, :8]
            score = [1.0 for _ in range(coords.shape[0])]
        if coords.shape[1] == 9:  # x1,y1,x2,y2,x3,y3,x4,y4,score
            coord = coords[:, :8]
            score = coords[:, 8]
        if coords.shape[1] == 5:  # x1,y1,x2,y2,score
            score = coords[:, 4]
            x1, y1, x2, y2 = np.split(coords[:, :4], 4, axis=1)
            coord = np.concatenate([x1, y1, x2, y1, x2, y2, x1, y2], axis=1)

        coords = coord.reshape([-1, 4, 2])
        confs = score
        canvas = canvas if canvas else self._canvas
        self._canvas = self.mold_vertext_on_image(canvas, coords, confs, color, prefix)

    def mold_vertext_on_image(self, image_np, coords, confs, color, prefix):
        """
        :param image_np:
        :param coords: batchx #vertext x 2
        :param content:
        :return:
        """
        num = coords.shape[0]
        if num == 0:
            return image_np
        if coords.shape[1] == 4:
            coords = np.array([order_points(coords[i]) for i in range(num)])
        vex_coords = np.reshape(coords, [num, -1, 2]).astype(np.int32)
        if prefix is not None:
            for i in range(num):
                loc = tuple(vex_coords[i][0])
                image_np = cv2ImgAddText(
                    image_np, prefix[i], loc[0], loc[1] - 20, (255, 0, 0), 1
                )
        for i in range(num):
            cv2.polylines(
                image_np, [vex_coords[i]], True, color=color, thickness=1, lineType=30
            )
        return image_np


class Box(object):
    def __init__(self, bndbox, text="", label_score=1.0, source=""):
        """
        Interface Class for Visualization tools
        Args:
            bndbox: list or array-like coords for bounding box, supporting both 2-point rep style:[x1, y1, x2, y2]
                   and 8-point rep style: [x1, y1, x2, y2, x3, y3, x4, y4]
            text(str): text to display above bounding box
            label_score(float): bounding box score
            source(str): box's source model, support visualization for multiple models' outputs
        """
        self.bndbox = bndbox
        self.text = text
        self.label_score = label_score
        self.source = source


def draw(image_path, box_list, color=None):
    """

    Args:
        image_path: BGR format numpy or image_path
        box_list(Box): list of Box object

    Returns:
        VisTool object, with .canvas property to access drawn image

    """
    vt = VisTool(image_path)
    sources = list(set([getattr(box, "source", "") for box in box_list]))
    palette = get_palette()
    colors = [hash(s) % palette.shape[0] for s in sources]
    for i, source in enumerate(sources):
        source_box_list = list(
            filter(lambda x: getattr(x, "source", "") == source, box_list)
        )
        coord = np.array([b.bndbox for b in source_box_list], dtype=np.float32)  # nx8
        scores = np.array(
            [getattr(b, "label_score", 1.0) for b in source_box_list], dtype=np.float32
        )  # n
        coords = np.concatenate([coord, np.expand_dims(scores, -1)], axis=1)
        display_info = [b.text for b in source_box_list]
        vt.draw(
            coords, color=colors[i] if color is None else color, prefix=display_info
        )
    return vt
