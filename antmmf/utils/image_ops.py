# Copyright (c) 2023 Ant Group and its affiliates.

import math
import random

import cv2
import numbers
import numpy as np
import os
import torch
import torchvision
from PIL import Image, ImageOps, ExifTags
from torchvision.transforms.functional import pad as img_pad
from torchvision.transforms.functional import resize as img_resize


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group, **kwargs):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group, **kwargs):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False, **kwargs):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(
                        ret[i]
                    )  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, **kwargs):
        """
        :param tensor: bsz, c, h, w / c, h, w
        :param kwargs:
        :return:
        """
        ndim = tensor.ndim
        if ndim == 3:
            tensor.unsqueeze_(0)
        num_channels = tensor.size(1)

        # align input channels
        if num_channels != len(self.mean):
            rep_mean = torch.tensor(self.mean * (num_channels // len(self.mean)))
            rep_std = torch.tensor(self.std * (num_channels // len(self.std)))

        else:
            rep_mean, rep_std = self.mean, self.std

        # align tensor shape
        rep_mean = torch.tensor(rep_mean).view(1, num_channels, 1, 1)
        rep_std = torch.tensor(rep_std).view(1, num_channels, 1, 1)

        # rep_mean.max()>1 for detectron2 preprocess:
        # https://detectron2.readthedocs.io/en/latest/modules/config.html
        # in which case, no need to normalize pixel value to [0, 1]
        if torch.max(tensor) > 1 and rep_mean.max() <= 1:
            tensor.div_(255.0)
        tensor.sub_(rep_mean).div_(rep_std)
        if ndim == 3:
            tensor.squeeze_(0)
        return tensor


class GroupScale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group, **kwargs):
        return [self.worker(img) for img in img_group]


class ImageLongsideScaleAndPad(object):
    """
    Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the longer edge. So that
    the image size is under control.
    For example, if height > width, then image will be
    rescaled to (size, size*width/height)

    size: size of the longer edge
    interpolation: Default: PIL.Image.BILINEAR

    Args:
        max_size (int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        pad (bool): Whether to pad image to (max_size, max_size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(
        self, max_size, random_scale=False, pad=False, interpolation=Image.BILINEAR
    ):
        assert isinstance(max_size, int)

        if random_scale is False:
            self.scales = [max_size]
        else:
            self.scales = [32 * i for i in range(7, 25) if 32 * i <= max_size]
            if max_size not in self.scales:
                self.scales.append(max_size)
        self.random_scale = random_scale
        self.interpolation = interpolation
        self.pad = pad

    def __call__(self, img, **kwargs):
        """
        Args:
            img (torch.tensor): Image to be scaled.

        Returns:
            torch.tensor: Rescaled image.
        """
        if self.random_scale:
            max_size = random.choice(self.scales)
        else:
            max_size = self.scales[-1]
        resized_imgs = img_resize(
            img, self.get_resize_size(img, max_size), self.interpolation
        )
        if self.pad is True:
            if isinstance(resized_imgs, torch.Tensor):
                h, w = resized_imgs.shape[-2:]
            else:
                w, h = resized_imgs.size
            h_padding, v_padding = self.max_size - w, self.max_size - h
            l_pad, t_pad = 0, 0
            r_pad, b_pad = h_padding, v_padding
            padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
            resized_imgs = img_pad(resized_imgs, padding, 0)
        return resized_imgs

    def get_resize_size(self, image, max_size):
        """
        Args:
            image: PIL Image or torch.tensor
            max_size:

        Returns:

        Note the height/width order difference
        >>> pil_img = Image.open("raw_img_tensor.jpg")
        >>> pil_img.size
        (640, 480)  # (width, height)
        >>> np_img = np.array(pil_img)
        >>> np_img.shape
        (480, 640, 3)  # (height, width, 3)
        """
        # note the order of height and width for different inputs
        if isinstance(image, torch.Tensor):
            # width, height = image.shape[-2:]
            height, width = image.shape[-2:]
        else:
            width, height = image.size

        if height >= width:
            ratio = width * 1.0 / height
            new_height = max_size
            new_width = new_height * ratio
        else:
            ratio = height * 1.0 / width
            new_width = max_size
            new_height = new_width * ratio
        size = (int(new_height), int(new_width))
        return size


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = (
            crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        )

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group, **kwargs):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h
        )
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == "L" and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)
        return oversample_group


class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = (
            crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        )

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group, **kwargs):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == "L" and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):
    def __init__(
        self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True
    ):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = (
            input_size if not isinstance(input_size, int) else [input_size, input_size]
        )
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group, **kwargs):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
            for img in crop_img_group
        ]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1]
            )

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group, **kwargs):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3.0 / 4, 4.0 / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group, **kwargs):
        if img_group[0].mode == "L":
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == "RGB":
            if self.roll:
                return np.concatenate(
                    [np.array(x)[:, :, ::-1] for x in img_group], axis=2
                )
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]"""

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic, **kwargs):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):
    def __call__(self, data, **kwargs):
        return data


class ImageLoader:
    def __init__(
        self,
        modality="RGB",
        root_path="../../data",
        image_tmpl="img_{:05d}.jpg",
        *args,
        **kwargs
    ):
        self.modality = modality
        self.root_path = root_path
        self.image_tmpl = image_tmpl

    def load_image(self, directory, idx, **kwargs):
        if self.modality == "RGB" or self.modality == "RGBDiff":
            try:
                if self.image_tmpl == "{}_{:05d}.jpg":
                    file_name = self.image_tmpl.format(os.path.basename(directory), idx)
                else:
                    file_name = self.image_tmpl.format(idx)
                return [
                    Image.open(
                        os.path.join(self.root_path, directory, file_name)
                    ).convert("RGB")
                ]
            except Exception:
                if self.image_tmpl == "{}_{:05d}.jpg":
                    file_name = self.image_tmpl.format(directory, 1)
                else:
                    file_name = self.image_tmpl.format(1)
                print(
                    "error loading image:",
                    os.path.join(self.root_path, directory, str(idx)),
                )
                return [
                    Image.open(
                        os.path.join(self.root_path, directory, file_name)
                    ).convert("RGB")
                ]
        elif self.modality == "Flow":
            if self.image_tmpl == "flow_{}_{:05d}.jpg":  # ucf
                x_img = Image.open(
                    os.path.join(
                        self.root_path, directory, self.image_tmpl.format("x", idx)
                    )
                ).convert("L")
                y_img = Image.open(
                    os.path.join(
                        self.root_path, directory, self.image_tmpl.format("y", idx)
                    )
                ).convert("L")
            elif self.image_tmpl == "{:06d}-{}_{:05d}.jpg":  # something v1 flow
                x_img = Image.open(
                    os.path.join(
                        self.root_path,
                        "{:06d}".format(int(directory)),
                        self.image_tmpl.format(int(directory), "x", idx),
                    )
                ).convert("L")
                y_img = Image.open(
                    os.path.join(
                        self.root_path,
                        "{:06d}".format(int(directory)),
                        self.image_tmpl.format(int(directory), "y", idx),
                    )
                ).convert("L")
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(
                        os.path.join(
                            self.root_path, directory, self.image_tmpl.format(idx)
                        )
                    ).convert("RGB")
                except Exception:
                    print(
                        "error loading flow file:",
                        os.path.join(
                            self.root_path, directory, self.image_tmpl.format(idx)
                        ),
                    )
                    flow = Image.open(
                        os.path.join(
                            self.root_path, directory, self.image_tmpl.format(1)
                        )
                    ).convert("RGB")
                # the input flow file is RGB image with (flow_x, flow_y, blank)
                # for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert("L")
                y_img = flow_y.convert("L")

            return [x_img, y_img]


class ExifImageLoader(object):
    """
    load image with exif info
    """

    @staticmethod
    def apply_exif_orientation(image):
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
            return image
        except Exception:
            return image

    @staticmethod
    def load_with_exif(file):
        image_data = Image.open(file)
        image_data = ExifImageLoader.apply_exif_orientation(image_data)
        if not image_data.mode == "RGB":
            image_data = image_data.convert("RGB")
        return image_data


class CV2ImageLoader(object):
    """
    faster than PIL loader
    """

    @staticmethod
    def load(file):
        img = cv2.imread(file)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img


class YoloImageLoader(object):
    def __init__(self, target_size, is_training=False):
        self.target_size = target_size
        self.is_training = is_training

    def __call__(self, img_path, **kwargs):
        img = cv2.imread(img_path)
        assert img is not None, "Image Not Found: " + img_path
        return self.resize_image_to_target(img)

    def resize_image_to_target(self, img):
        """
        Args:
            img: opencv BGR image
        Returns:
            image with longer side matching target_size
        """
        h0, w0 = img.shape[:2]  # orig hw
        # resize image longer side to target_size
        ratio = self.target_size / max(h0, w0)
        resize_to = (int(w0 * ratio), int(h0 * ratio))
        if (
            ratio != 1
        ):  # always resize down, only resize up if training with augmentation
            interp = (
                cv2.INTER_AREA
                if ratio < 1 and not self.is_training
                else cv2.INTER_LINEAR
            )
            img = cv2.resize(img, resize_to, interpolation=interp)
        # img, hw_original, hw_resized
        return {"image": img, "origin_size": (h0, w0), "image_size": img.shape[:2]}


if __name__ == "__main__":
    trans = torchvision.transforms.Compose(
        [
            GroupScale(256),
            GroupRandomCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    im = Image.open("../tensorflow-model-zoo.torch/lena_299.png")

    color_group = [im] * 3
    rst = trans(color_group)

    gray_group = [im.convert("L")] * 9
    gray_rst = trans(gray_group)

    trans2 = torchvision.transforms.Compose(
        [
            GroupRandomSizedCrop(256),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print(trans2(color_group))
