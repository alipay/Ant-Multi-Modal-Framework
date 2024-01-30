# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

"""
Boxes for object detection task, the code is copied from:
https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/boxes.py
"""

import math
import torch
import numpy as np
from enum import Enum
from typing import List, Tuple, Union

from antmmf.structures import SizedDataStructure


class IoUType(Enum):
    Normal = 1
    GIoU = 2
    DIoU = 3
    CIoU = 4


class Boxes(SizedDataStructure):
    """
    This structure stores a list of boxes as a Nx4 :external:py:class:`Tensor <torch.Tensor>`.
    It supports some common methods about boxes (`area`, `clip`, `nonempty`, etc),

    Args:
        tensor (torch.Tensor): a Nx4 matrix.  Each row is (x1, y1, x2, y2) or (x1, y1, w, h).
        box_mode (int): Boxes.BOX_MODE_XYXY, Boxes.BOX_MODE_XYWH or BOX_MODE_CXCYWH
    """

    BOX_MODE_XYXY = 0
    BOX_MODE_XYWH = 1
    BOX_MODE_CXCYWH = 2

    def __init__(
        self, tensor: Union[torch.Tensor, np.ndarray], box_mode: int = BOX_MODE_XYXY
    ):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)

        assert tensor.dim() == 2 and tensor.shape[-1] == 4, tensor.size()
        if box_mode not in [
            Boxes.BOX_MODE_XYXY,
            Boxes.BOX_MODE_XYWH,
            Boxes.BOX_MODE_CXCYWH,
        ]:
            raise ValueError(
                "Only Boxes.BOX_MODE_XYXY and Boxes.BOX_MODE_XYWH are available for box_mode"
            )

        super(Boxes, self).__init__(tensor, box_mode=box_mode)

    @property
    def box_mode(self):
        return self.kwargs.get("box_mode", Boxes.BOX_MODE_XYXY)

    @box_mode.setter
    def box_mode(self, box_mode: int):
        if box_mode not in [
            Boxes.BOX_MODE_XYXY,
            Boxes.BOX_MODE_XYWH,
            Boxes.BOX_MODE_CXCYWH,
        ]:
            raise ValueError(
                "Only Boxes.BOX_MODE_XYXY and Boxes.BOX_MODE_XYWH are available for box_mode"
            )
        self.kwargs["box_mode"] = box_mode

    def _box_mode_keeper(self):
        """
        In some inplace operations, we may change the box_mode of Boxes, this mode keeper context will help
        us to keep the box_mode unchanged.

        Usage::

            with self._box_mode_keeper():
                ...
        """
        box = self

        class Keeper:
            def __enter__(self):
                self.box_mode = box.box_mode

            def __exit__(self, exc_type, exc_val, exc_tb):
                box.convert_box_mode(self.box_mode)

        return Keeper()

    def _convert_to_xywh(self):
        if self.box_mode == Boxes.BOX_MODE_XYWH:
            return
        if self.box_mode == Boxes.BOX_MODE_XYXY:
            x1, y1, x2, y2 = (
                self.tensor[:, 0],
                self.tensor[:, 1],
                self.tensor[:, 2],
                self.tensor[:, 3],
            )
            w, h = x2 - x1, y2 - y1
            self.tensor = torch.stack((x1, y1, w, h), dim=1)
            self.box_mode = Boxes.BOX_MODE_XYWH
            return
        if self.box_mode == Boxes.BOX_MODE_CXCYWH:
            cx, cy, w, h = (
                self.tensor[:, 0],
                self.tensor[:, 1],
                self.tensor[:, 2],
                self.tensor[:, 3],
            )
            self.tensor = torch.stack((cx - w * 0.5, cy - h * 0.5, w, h), dim=1)
            self.box_mode = Boxes.BOX_MODE_XYWH

    def convert_box_mode(self, box_mode: int):
        """
        Convert mode of the boxes, for example, convert the mode of boxes from xyxy to xywh, vice versa.

        Args:
            box_mode (int): only Boxes.BOX_MODE_XYXY and Boxes.BOX_MODE_XYWH are available.

        Returns:
            self (Boxes):
        """
        if box_mode not in [
            Boxes.BOX_MODE_XYXY,
            Boxes.BOX_MODE_XYWH,
            Boxes.BOX_MODE_CXCYWH,
        ]:
            raise ValueError(
                "Only Boxes.BOX_MODE_XYXY and Boxes.BOX_MODE_XYWH are available for box_mode"
            )

        if self.box_mode == box_mode:
            return self

        self._convert_to_xywh()
        if box_mode == Boxes.BOX_MODE_XYXY:
            x1, y1, w, h = (
                self.tensor[:, 0],
                self.tensor[:, 1],
                self.tensor[:, 2],
                self.tensor[:, 3],
            )
            x2, y2 = x1 + w, y1 + h
            self.tensor = torch.stack((x1, y1, x2, y2), dim=1)
        if box_mode == Boxes.BOX_MODE_CXCYWH:
            x1, y1, w, h = (
                self.tensor[:, 0],
                self.tensor[:, 1],
                self.tensor[:, 2],
                self.tensor[:, 3],
            )
            self.tensor = torch.stack((x1 + w * 0.5, y1 + h * 0.5, w, h), dim=1)

        self.box_mode = box_mode
        return self

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        with self._box_mode_keeper():
            self.convert_box_mode(Boxes.BOX_MODE_XYXY)
            box = self.tensor
            area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size

        with self._box_mode_keeper():
            self.convert_box_mode(Boxes.BOX_MODE_XYXY)
            x1 = self.tensor[:, 0].clamp(min=0, max=w)
            y1 = self.tensor[:, 1].clamp(min=0, max=h)
            x2 = self.tensor[:, 2].clamp(min=0, max=w)
            y2 = self.tensor[:, 3].clamp(min=0, max=h)
            self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            torch.Tensor: a binary vector which represents whether each box is empty (False) or non-empty (True).
        """
        with self._box_mode_keeper():
            box = self.convert_box_mode(Boxes.BOX_MODE_XYWH)
            keep = (box[:, 2] > threshold) & (box[:, 3] > threshold)
        return keep

    def inside_box(
        self, box_size: Tuple[int, int], boundary_threshold: int = 0
    ) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            torch.Tensor: a binary vector, indicating whether each box is inside the reference box.
        """
        with self._box_mode_keeper():
            self.convert_box_mode(Boxes.BOX_MODE_XYXY)
            height, width = box_size
            inside_indices = (
                (self.tensor[..., 0] >= -boundary_threshold)
                & (self.tensor[..., 1] >= -boundary_threshold)
                & (self.tensor[..., 2] < width + boundary_threshold)
                & (self.tensor[..., 3] < height + boundary_threshold)
            )
        return inside_indices

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The box centers in a Nx2 array of (x, y).
        """
        with self._box_mode_keeper():
            self.convert_box_mode(Boxes.BOX_MODE_CXCYWH)
            centers = self.tensor[:, :2]
        return centers

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Args:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        assert (
            len(set([box.box_mode for box in boxes_list])) == 1
        ), "All boxes' box_mode in boxes_list must be same"
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    def _pairwise_intersection(self, other: "Boxes") -> torch.Tensor:
        """
        Assume the size of self is N, and given another boxes of size M,
        compute the intersection area between __all__ N x M pairs of boxes.
        The box order must be (x_min, y_min, x_max, y_max)

        .. warning::

            Do not call this function in other place, because it is not safe since
            it is only implemented under the assumption of the box_mode of the
            Boxes is Boxes.BOX_MODE_XYXY.

        Args:
            other (Boxes): a `Boxes`. Contains M boxes.

        Returns:
            torch.Tensor: intersection, sized [N,M].
        """
        boxes1, boxes2 = self.tensor, other.tensor
        width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
            boxes1[:, None, :2], boxes2[:, :2]
        )  # [N,M,2]

        width_height.clamp_(min=0)  # [N,M,2]
        intersection = width_height.prod(dim=2)  # [N,M]
        return intersection

    def pairwise_iou(
        self, other: "Boxes", iou_type: Enum = IoUType.Normal
    ) -> torch.Tensor:
        """
        Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
        with slight modifications.

        Assume the size of self is N, and given another boxes of size M,, compute the IoU
        (intersection over union) between **all** N x M pairs of boxes.
        The box order must be (x_min, y_min, x_max, y_max).

        Args:
            boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

        Returns:
            torch.Tensor: IoU, sized [N,M].
        """
        modes = [self.box_mode, other.box_mode]
        self.convert_box_mode(Boxes.BOX_MODE_XYXY)
        other = other.convert_box_mode(Boxes.BOX_MODE_XYXY)

        area1 = self.area()  # [N]
        area2 = other.area()  # [M]
        inter = self._pairwise_intersection(other)  # [N, M]

        # handle empty boxes
        iou = torch.zeros_like(inter)
        # avoid division by 0.
        non_zero_mask = inter > 0.0
        iou[non_zero_mask] = (
            inter[non_zero_mask] / (area1[:, None] + area2 - inter)[non_zero_mask]
        )

        x11, y11, x12, y12 = self.tensor.unbind(1)
        x21, y21, x22, y22 = other.tensor.unbind(1)
        self.convert_box_mode(modes[0])
        other.convert_box_mode(modes[1])

        if iou_type is IoUType.Normal:
            return iou

        cw = torch.max(x12.view(-1, 1), x22) - torch.min(x11.view(-1, 1), x21)  # [N, M]
        ch = torch.max(y12.view(-1, 1), y22) - torch.min(y11.view(-1, 1), y21)  # [N, M]

        if iou_type is IoUType.GIoU:
            c_area = cw * ch
            union = area1.view(-1, 1) + area2 - inter
            iou = torch.where(c_area > 0.0, iou - (c_area - union) / c_area, iou)
            return iou

        c2 = cw**2 + ch**2
        rho2 = (
            ((x11 + x12).view(-1, 1) - (x21 + x22)) ** 2
            + ((y11 + y12).view(-1, 1) - (y21 + y22)) ** 2
        ) / 4.0
        if iou_type is IoUType.DIoU:
            iou = torch.where(c2 > 0.0, iou - rho2 / c2, iou)
            return iou

        if iou_type is IoUType.CIoU:
            w1, h1 = x12 - x11, y12 - y11
            w2, h2 = x22 - x21, y22 - y21
            v = (4 / math.pi**2) * torch.pow(
                torch.atan(w1 / h1).view(-1, 1) - torch.atan(w2 / h2), 2
            )
            with torch.no_grad():
                alpha = v / (1 + v - iou)
            iou = torch.where(
                (c2 > 0.0) & (iou < 1.0), iou - rho2 / c2 - v * alpha, iou
            )
            return iou

        raise NotImplementedError(f"{iou_type} is not supported.")

    def matched_pairwise_iou(
        self, other: "Boxes", iou_type: Enum = IoUType.Normal
    ) -> torch.Tensor:
        """
        Compute pairwise intersection over union (IOU) of two sets of matched
        boxes that have the same number of boxes.
        Similar to :func:`Boxes.pairwise_iou`, but computes only diagonal elements of the matrix.

        Args:
            other (Boxes): bounding boxes, sized [N,4].

        Returns:
            torch.Tensor: iou, sized [N].
        """
        assert len(self) == len(
            other
        ), f"boxes should have the same number of entries, got {len(self)}, {len(other)}"
        modes = [self.box_mode, other.box_mode]
        self.convert_box_mode(Boxes.BOX_MODE_XYXY)
        other = other.convert_box_mode(Boxes.BOX_MODE_XYXY)

        area1 = self.area()  # [N]
        area2 = other.area()  # [N]
        box1, box2 = self.tensor, other.tensor
        lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
        rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
        wh = (rb - lt).clamp(min=0)  # [N,2]
        inter = wh[:, 0] * wh[:, 1]  # [N]

        # handle empty boxes
        iou = torch.zeros_like(inter)
        # avoid division by 0.
        non_zero_mask = inter > 0.0
        iou[non_zero_mask] = (
            inter[non_zero_mask] / (area1 + area2 - inter)[non_zero_mask]
        )

        x11, y11, x12, y12 = self.tensor.unbind(1)
        x21, y21, x22, y22 = other.tensor.unbind(1)
        self.convert_box_mode(modes[0])
        other.convert_box_mode(modes[1])

        if iou_type is IoUType.Normal:
            return iou

        cw = torch.max(x12, x22) - torch.min(x11, x21)  # [N, M]
        ch = torch.max(y12, y22) - torch.min(y11, y21)  # [N, M]

        if iou_type is IoUType.GIoU:
            c_area = cw * ch
            union = area1 + area2 - inter
            iou = torch.where(c_area > 0.0, iou - (c_area - union) / c_area, iou)
            return iou

        c2 = cw**2 + ch**2
        rho2 = ((x11 + x12 - x21 - x22) ** 2 + (y11 + y12 - y21 - y22) ** 2) / 4.0
        if iou_type is IoUType.DIoU:
            iou = torch.where(c2 > 0.0, iou - rho2 / c2, iou)
            return iou

        if iou_type is IoUType.CIoU:
            w1, h1 = x12 - x11, y12 - y11
            w2, h2 = x22 - x21, y22 - y21
            v = (4 / math.pi**2) * torch.pow(
                torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
            )
            with torch.no_grad():
                alpha = v / (1 + v - iou)
            iou = torch.where(
                (c2 > 0.0) & (iou < 1.0), iou - rho2 / c2 - v * alpha, iou
            )
            return iou

        raise NotImplementedError(f"{iou_type} is not supported.")
