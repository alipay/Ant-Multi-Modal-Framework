# Copyright (c) 2023 Ant Group and its affiliates.
import glob
import math
import os.path as osp
import random
import warnings

import PIL
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageFont, ImageDraw

from antmmf.structures.sample import Sample
from antmmf.utils.general import get_antmmf_root
from antmmf.utils.image_ops import CV2ImageLoader


def build_bbox_tensors(infos, max_length):
    num_bbox = min(max_length, len(infos))

    # After num_bbox, everything else should be zero
    coord_tensor = torch.zeros((max_length, 4), dtype=torch.float)
    width_tensor = torch.zeros(max_length, dtype=torch.float)
    height_tensor = torch.zeros(max_length, dtype=torch.float)
    bbox_types = ["xyxy"] * max_length

    infos = infos[:num_bbox]
    sample = Sample()

    for idx, info in enumerate(infos):
        bbox = info["bounding_box"]
        x = bbox["top_left_x"]
        y = bbox["top_left_y"]
        width = bbox["width"]
        height = bbox["height"]

        coord_tensor[idx][0] = x
        coord_tensor[idx][1] = y
        coord_tensor[idx][2] = x + width
        coord_tensor[idx][3] = y + height

        width_tensor[idx] = width
        height_tensor[idx] = height
    sample.coordinates = coord_tensor
    sample.width = width_tensor
    sample.height = height_tensor
    sample.bbox_types = bbox_types

    return sample


def random_crop(img, four_side_ratios=[0.2, 0.1, 0.05], **kwargs):
    """
    Randomly crop the four sides of the image at different scales
    """
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    left20, right20, top20, bottom20 = (
        random.randint(0, int(w * four_side_ratios[0])),
        random.randint(0, int(w * four_side_ratios[0])),
        random.randint(0, int(h * four_side_ratios[0])),
        random.randint(0, int(h * four_side_ratios[0])),
    )
    left10, right10, top10, bottom10 = (
        random.randint(0, int(w * four_side_ratios[1])),
        random.randint(0, int(w * four_side_ratios[1])),
        random.randint(0, int(h * four_side_ratios[1])),
        random.randint(0, int(h * four_side_ratios[1])),
    )
    left5, right5, top5, bottom5 = (
        random.randint(0, int(w * four_side_ratios[2])),
        random.randint(0, int(w * four_side_ratios[2])),
        random.randint(0, int(h * four_side_ratios[2])),
        random.randint(0, int(h * four_side_ratios[2])),
    )
    idx = random.randint(0, 6)
    if idx == 0:
        return img[
            top20:h,
        ]
    elif idx == 1:
        return img[
            0 : h - bottom20,
        ]
    elif idx == 2:
        return img[:, left20:]
    elif idx == 3:
        return img[:, 0 : w - right20]
    elif idx == 4:
        return img[
            top10 : h - bottom10,
        ]
    elif idx == 5:
        return img[:, left10 : w - right10]
    elif idx == 6:
        return img[top5 : h - bottom5, left5 : w - right5]


def random_black(img, four_side_ratios=[0.2, 0.1, 0.05], **kwargs):
    """
    Randomly fill black color to  the four sides of the image at different scales
    """
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    idx = random.randint(0, 6)
    left20, right20, top20, bottom20 = (
        random.randint(0, int(w * four_side_ratios[0])),
        random.randint(0, int(w * four_side_ratios[0])),
        random.randint(0, int(h * four_side_ratios[0])),
        random.randint(0, int(h * four_side_ratios[0])),
    )
    left10, right10, top10, bottom10 = (
        random.randint(0, int(w * four_side_ratios[1])),
        random.randint(0, int(w * four_side_ratios[1])),
        random.randint(0, int(h * four_side_ratios[1])),
        random.randint(0, int(h * four_side_ratios[1])),
    )
    left5, right5, top5, bottom5 = (
        random.randint(0, int(w * four_side_ratios[2])),
        random.randint(0, int(w * four_side_ratios[2])),
        random.randint(0, int(h * four_side_ratios[2])),
        random.randint(0, int(h * four_side_ratios[2])),
    )
    if idx == 0:
        img[0:top20, :] = 0
        return img
    elif idx == 1:
        img[h - bottom20 : h, :] = 0
        return img
    elif idx == 2:
        img[:, 0:left20] = 0
        return img
    elif idx == 3:
        img[:, w - right20 : w] = 0
        return img
    elif idx == 4:
        img[0:top10, :] = 0
        img[h - bottom10 : h, :] = 0
        return img
    elif idx == 5:
        img[:, 0:left10] = 0
        img[:, w - right10 : w] = 0
        return img
    elif idx == 6:
        img[0:top5, :] = 0
        img[h - top5 : h, :] = 0
        img[:, 0:left5] = 0
        img[:, w - right5 : w] = 0
        return img


def random_flip(img, **kwargs):
    """
    Randomly flip image  around
    """
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_affine(img, **kwargs):
    """
    Random affine transformation, position, angle, scale
    """
    hori, vert = 0.1 * random.randint(2, 6), 0.1 * random.randint(2, 6)
    prob_h, prob_v = random.random(), random.random()
    ang = random.choice([10, 20, 90, 270])
    idx = random.randint(0, 3)
    if idx == 0:
        trans = T.RandomAffine(
            degrees=ang,
            translate=None,
            scale=None,
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=0,
        )
    elif idx == 2:
        trans = T.RandomAffine(
            degrees=(0, 0),
            translate=(hori, vert),
            scale=None,
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=0,
        )
    else:
        trans = T.RandomAffine(
            degrees=(0, 0),
            translate=None,
            scale=(0.5, 0.5),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=0,
        )
    img = trans(img)
    return img


def random_color(img, **kwargs):
    brightness, contrast, saturation, hue = (
        random.uniform(0, 0.5),
        random.uniform(0, 0.5),
        random.uniform(0, 0.5),
        random.uniform(0, 0.5),
    )
    trans = T.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )
    img = trans(img)
    return img


def get_low_clip(img):
    if img.min() < 0:
        low_clip = -1.0
    else:
        low_clip = 0.0
    return low_clip


def poisson_noise(img):
    """
    Add poisson noise to img.
    @param img: ndarray,
           Input image data on range[0,255].
    @returns out : ndarray
           Output interger image data on range [0, 255]
    """
    image = np.array(img / 255.0, dtype=float)
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    low_clip = get_low_clip(image)
    if low_clip == -1.0:
        old_max = image.max()
        image = (image + 1.0) / (old_max + 1.0)
    out = np.random.poisson(image * vals) / float(vals)
    if low_clip == -1.0:
        out = out * (old_max + 1.0) - 1.0
    low_clip = get_low_clip(out)
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def salt_pepper_noise(img, amount=0.05, salt_vs_pepper=0.5):
    """
    Add salt noise, pepper noise, or, s&p noise to img.
    @param img: ndarray
           Input image data on range[0,255].
    @param amount : float, optional
           Proportion of image pixels to replace with noise on range [0, 1].
           Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.05
    @param salt_vs_pepper:float, optional
           Proportion of salt vs. pepper noise for 's&p' on range [0, 1].
           Higher values represent more salt. Default : 0.5 (equal amounts)
    @returns out : ndarray
           Output interger image data on range [0, 255]
    """
    image = np.array(img / 255.0, dtype=float)
    out = image.copy()
    low_clip = get_low_clip(image)
    p = amount
    q = salt_vs_pepper
    flipped = np.random.choice([True, False], size=image.shape, p=[p, 1 - p])
    salted = np.random.choice([True, False], size=image.shape, p=[q, 1 - q])
    peppered = ~salted
    out[flipped & salted] = 1
    out[flipped & peppered] = low_clip
    low_clip = get_low_clip(out)
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def gaussian_speckle_noise(
    img, mean=0.0, var=0.01, noise_type="gaussian", img_channels=[0, 1, 2]
):
    """
    Add gaussian noise or speckle noise to img.
    @param img: ndarray
           Input image data on range[0,255].
    @param mean:float, optional
           Mean of random distribution. Used in 'gaussian' and 'speckle'.Default : 0.
    @param var:float, optional
           Variance of random distribution. Used in 'gaussian' and 'speckle'.
           Note: variance = (standard deviation) ** 2. Default : 0.01
    @param img_channels: list, in allowed_img_channels
           Img_channels, to which the noise is really added.
    @param noise_type: 'gaussion' or 'speckle'
    @returns out : ndarray
           Output interger image data on range [0, 255]
    """
    image = np.array(img / 255.0, dtype=float)
    out = image.copy()
    noise = np.random.normal(mean, var**0.5, image.shape)
    if len(image.shape) == 2:
        out = image + noise
    else:
        for channel in [0, 1, 2]:
            if channel in img_channels:
                if noise_type == "speckle":
                    out[:, :, channel] = (
                        image[:, :, channel]
                        + image[:, :, channel] * noise[:, :, channel]
                    )
                else:  # 'gaussian'
                    out[:, :, channel] = image[:, :, channel] + noise[:, :, channel]
    low_clip = get_low_clip(out)
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def random_noise(img, **kwargs):
    """
    Add random noise types to one image.
    @params random_type: string  in allowed_noise_types
            - 'gaussian'  Gaussian-distributed additive noise.
            - 'poisson'   Poisson-distributed noise generated from the data.
            - 'salt'      Replaces random pixels with 255.
            - 'pepper'    Replaces random pixels with 0.
            - 's&p'       Replaces random pixels with 255 or 0.
            - 'speckle'   Multiplicative noise using out = image + n*image, where
                          n is uniform noise with specified mean & variance.
    """
    allowed_noise_types = ["gaussian", "poisson", "salt", "pepper", "s&p", "speckle"]
    allowed_img_channels = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    random_type = random.sample(allowed_noise_types, 1)[0]
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if random_type == "gaussian":
        mean = random.random()
        var = random.random()
        img_channels = random.sample(allowed_img_channels, 1)[0]
        noised_img = gaussian_speckle_noise(img, mean, var, "gaussian", img_channels)
    elif random_type == "s&p":
        amount = random.random()  # 0.05
        salt_vs_pepper = random.random()  # 0.5
        noised_img = salt_pepper_noise(img, amount, salt_vs_pepper)
    elif random_type == "salt":
        amount = random.random()  # 0.05
        salt_vs_pepper = 1.0
        noised_img = salt_pepper_noise(img, amount, salt_vs_pepper)
    elif random_type == "pepper":
        amount = random.random()  # 0.05
        salt_vs_pepper = 0.0
        noised_img = salt_pepper_noise(img, amount, salt_vs_pepper)
    elif random_type == "poisson":
        noised_img = poisson_noise(img)
    elif random_type == "speckle":
        mean = random.random()
        var = random.random()
        img_channels = random.sample(allowed_img_channels, 1)[0]
        noised_img = gaussian_speckle_noise(img, mean, var, "speckle", img_channels)
    else:
        noised_img = img
    noised_img = np.array(cv2.cvtColor(noised_img, cv2.COLOR_BGR2RGB))
    noised_img = np.where(noised_img > 255, 255, noised_img)
    noised_img = np.where(noised_img < 0, 0, noised_img)
    return noised_img


def gaussian_blur(img, ksize, sigma=0):
    """
    Gaussian Blurring, a Gaussian kernel is used.
    @param ksize: the kernel size of gaussian kernel,
            here the width and the height of the kernel is the same.
            ksize must be a positive odd integer.
    @param sigma: the standard deviation in the X direction,
            here sigma is set to 0, which means that it will be calculated automatically.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_blur(img, ksize):
    """
    Median Blurring, which takes the median of all the pixels under the kernel area
              and the central element is replaced with this median value.
    @param ksize : the kernel size, which must be a positive odd integer.
    """
    return cv2.medianBlur(img, ksize)


def bilateral_blur(img, d, sigma_color, sigma_space):
    """
    Bilateral Filtering, which takes a Gaussian filter in space,
           but one more Gaussian filter which is a function of pixel difference.
    @param d: diameter of each pixel neighborhood that is used during filtering.
           if d=-1, then it will be calculated automatically.
    @param sigma_color: standard deviation in color.
    @param sigma_space: standard deviation in space.
            where sigma_color are often set to the same as sigma_space.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def average_blur(img, ksize):
    """
    Averageing: blur is done by convolving an image with a normalized box filter.
    @param ksize: the kernel size of the boxFilter, which must be a positive odd integer.
    """
    return cv2.blur(img, (ksize, ksize))


def random_blur(img, **kwargs):
    """
    Add random blur types to one image.
    @params random_type: string  in allowed_blur_types
            - 'gaussian'
    """
    allowed_blur_types = ["gaussian", "median", "bilateral", "average"]
    random_type = random.sample(allowed_blur_types, 1)[0]
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[0], img.shape[1]
    # max_ksize = min(h,w)/20
    max_ksize = 50
    if random_type == "gaussian":
        ksize = random.randrange(3, max_ksize, 2)
        blurred_img = gaussian_blur(img, ksize)
    elif random_type == "median":
        ksize = random.randrange(3, max_ksize, 2)
        blurred_img = median_blur(img, ksize)
    elif random_type == "average":
        ksize = random.randrange(3, max_ksize, 2)
        blurred_img = average_blur(img, ksize)
    elif random_type == "bilateral":
        ksize = random.randrange(-1, 25, 2)
        sigma_color = random.randint(10, 150)
        sigma_space = sigma_color
        blurred_img = bilateral_blur(img, ksize, sigma_color, sigma_space)
    else:
        blurred_img = img
    blurred_img = np.array(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
    return blurred_img


def single_gray(img, channel=0):
    """
    Single channel value as output gray value. gray = single(r,g,b).
    @param img:ndarray,Input image data on range[0,255].
    @param channel: int, [0,1,2]
    @return out: ndarray,Output interger image data on range [0, 255].
    """
    image = np.array(img)
    h, w = image.shape[0], image.shape[1]
    out = np.zeros((h, w, 3), dtype=image.dtype)
    if len(image.shape) == 2:
        out[:, :, 0] = image
        out[:, :, 1] = image
        out[:, :, 2] = image
    else:
        out[:, :, 0] = image[:, :, channel]
        out[:, :, 1] = image[:, :, channel]
        out[:, :, 2] = image[:, :, channel]
    out = np.uint8(out)
    return out


def multi_gray(img, gray_type="average"):
    """
    Get output gray_value from multiple-channels.
    @param img:ndarray,Input image data on range[0,255].
    @param gray_type: string, ['average','maximum','weighted_sum']
    @return out: ndarray,Output interger image data on range [0, 255].
    """
    image = np.array(img)
    h, w = image.shape[0], image.shape[1]
    out = np.zeros((h, w, 3), dtype=image.dtype)
    if len(image.shape) == 2:
        out[:, :, 0] = image
        out[:, :, 1] = image
        out[:, :, 2] = image
    else:
        if gray_type == "average":
            im2 = np.sum(image, axis=2)
            out[:, :, 0] = im2 / 3.0
            out[:, :, 1] = im2 / 3.0
            out[:, :, 2] = im2 / 3.0
        elif gray_type == "maximum":
            out[:, :, 0] = np.max(image, axis=2)
            out[:, :, 1] = np.max(image, axis=2)
            out[:, :, 2] = np.max(image, axis=2)
        else:  # weighted_sum: Gray(i,j) = 0.299 * R(i,j) + 0.587 * G(i,j) + 0.114 * B(i,j)
            out[:, :, 0] = image[:, :, 0] * 0.299
            out[:, :, 1] = image[:, :, 1] * 0.587
            out[:, :, 2] = image[:, :, 2] * 0.114
            # out = np.sum(image,axis=2)
    out = np.uint8(out)
    return out


def random_gray(img, **kwargs):
    """
    Add random gray types to one image.
    @params random_type: string  in allowed_gray_types
           - 'single': single_channle as output gray value. gray = single(r,g,b)
           - 'average': average value of three channels, gray = (r+g+b)/3.0
           - 'maximum': max value of three channels, gray = max(r,g,b)
           - 'weighted_sum': 0.299 * R(i,j) + 0.587 * G(i,j) + 0.114 * B(i,j)
    """
    allowed_gray_types = ["single", "average", "maximum", "weighted_sum"]
    random_type = random.sample(allowed_gray_types, 1)[0]
    if random_type == "single":
        single_channel = random.randint(0, 2)
        out = single_gray(img, single_channel)
    else:
        out = multi_gray(img, random_type)

    return out


def pet_custom_transform(img, idx=None):
    """
    Randomly transform the image, 0 means the original image
    """
    if idx is None:
        idx = random.randint(0, 8)
    if idx == 1:
        img = random_crop(img)
    elif idx == 2:
        img = random_black(img)
    elif idx == 3:
        img = random_flip(img)
    elif idx == 4:
        img = random_affine(img)
    elif idx == 5:
        img = random_color(img)
    elif idx == 6:
        img = random_noise(img)
    elif idx == 7:
        img = random_blur(img)
    elif idx == 8:
        img = random_gray(img)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img, mode="RGB")
    return img, idx


def pictureInPicture(tmp_img, picture_dir=None, **kwargs):
    if not picture_dir:
        warnings.warn("skip randLogo augmentation for available logo paths not set ")
        return

    base_paths = glob.glob(picture_dir + "/*")
    while True:
        base_img = CV2ImageLoader.load(random.choice(base_paths))
        if min(base_img.size) > min(tmp_img.size) / 1.5:
            break
    COLOR_LIST = [
        "Pink",
        "Purple",
        "Blue",
        "Green",
        "Orange",
        "White",
        "LightPink",
        "PaleVioletRed",
        "Lavender",
        "LightSkyBlue",
        "Yellow",
        "Gold",
        "NavajoWhite",
        "MediumSpringGreen",
        "Beige",
        "LemonChiffon",
        "Cyan",
        "Turquoise",
        "PeachPuff",
        "Red",
        "Black",
    ]
    rand_bg = random.uniform(0, 1)
    blur_filter = ImageFilter.GaussianBlur(radius=random.randint(20, 30))
    if rand_bg <= 0.4:
        # 背景模糊
        if random.uniform(0, 1) <= 0.5:
            # 背景模糊
            base_img = base_img.filter(blur_filter)
        else:
            # 前景模糊作为背景
            if isinstance(tmp_img, list):
                base_img = tmp_img[0].filter(blur_filter)
            else:
                base_img = tmp_img.filter(blur_filter)
    elif rand_bg > 0.4 and rand_bg < 0.8:
        # 纯色背景
        base_img = Image.new(
            "RGB",
            [base_img.size[0], base_img.size[1]],
            COLOR_LIST[random.randint(0, len(COLOR_LIST) - 1)],
        )
    else:
        # 图片背景
        base_img = base_img

    # 非拼图
    [base_w, base_h] = base_img.size[0:2]
    [tmp_w, tmp_h] = tmp_img.size[0:2]

    min_base = np.min([base_w, base_h])
    max_tmp = np.max([tmp_w, tmp_h])
    paste_ratio = random.uniform(0.8, 1) * min_base / max_tmp
    [tmp_w, tmp_h] = [int(tmp_w * paste_ratio), int(tmp_h * paste_ratio)]

    corx_start = int(
        random.uniform(
            math.floor((base_w - tmp_w) / 4), math.floor((base_w - tmp_w) / 2)
        )
    )
    cory_start = int(
        random.uniform(
            math.floor((base_h - tmp_h) / 4), math.floor((base_h - tmp_h) / 2)
        )
    )

    if random.uniform(0, 1) <= 0.5:
        box = (
            corx_start,
            cory_start,
            corx_start + tmp_w,
            cory_start + tmp_h,
        )  # 底图上需要P掉的区域
    else:
        box = (
            base_w - corx_start - tmp_w,
            base_h - cory_start - tmp_h,
            base_w - corx_start,
            base_h - cory_start,
        )

    if np.any(np.array(box) < 0):
        print(box)
    # 使用 paste(region, box) 方法将图片粘贴到另一种图片上去.
    region = tmp_img.resize((box[2] - box[0], box[3] - box[1]))
    base_img.paste(region, box)
    collage_img = base_img
    return collage_img


def randText(img, **kwargs):  # done
    temp_image = Image.fromarray(np.copy(img))
    h, w = temp_image.size
    left, top1, top2 = (
        random.randint(0, int(0.1 * w)),
        random.randint(0, int(0.1 * h)),
        random.randint(0, int(0.9 * h)),
    )
    position = (left, top1)
    if random.random() > 0.5:
        position = (left, top2)
    font_path = osp.join(get_antmmf_root(), "utils/visual_utils/huawenfangsong.ttf")
    font = ImageFont.truetype(font_path, 30, encoding="utf-8")

    from faker import Faker

    fake = Faker("zh_CN")
    raw_str = fake.text(max_nb_chars=100)

    l = len(raw_str)
    start = random.randint(0, l - 11)
    ss = raw_str[start : start + 10]
    draw = ImageDraw.Draw(temp_image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]
    color = random.choice(colors)
    draw.text(position, ss, font=font, fill=color)
    return temp_image


def randLogo(img, random_logo_dir=None, **kwargs):
    if not random_logo_dir:
        warnings.warn("skip randLogo augmentation for available logo paths not set ")
        return
    random_logo_paths = glob.glob(random_logo_dir + "/*")
    img_pip = Image.fromarray(np.copy(img))
    logo_path = random.choice(random_logo_paths)
    logo_img = Image.open(logo_path).convert("RGBA")
    [size0, size1] = img_pip.size
    cordx_start = random.randint(0, size0)
    cordy_start = random.randint(0, size1)
    [logo_size0, logo_size1] = logo_img.size[0:2]
    if np.min([logo_size0, logo_size1]) / np.min(img_pip.size[0:2]) > 0.3:
        logo_size1 = int(0.3 * cordx_start * logo_size1 / logo_size0)
        logo_size0 = int(0.3 * cordx_start)
    box = (cordx_start, cordy_start, cordx_start + logo_size0, cordy_start + logo_size1)
    while box[2] - box[0] == 0 or box[3] - box[1] == 0:
        logo_path = random.choice(random_logo_paths)
        logo_img = Image.open(logo_path).convert("RGBA")
        [size0, size1] = img_pip.size[0:2]
        cordx_start = random.randint(0, size0)
        cordy_start = random.randint(0, size1)
        [logo_size0, logo_size1] = logo_img.size[0:2]
        if np.min([logo_size0, logo_size1]) / np.min(img_pip.size[0:2]) > 0.3:
            logo_size1 = int(0.3 * cordx_start * logo_size1 / logo_size0)
            logo_size0 = int(0.3 * cordx_start)
        box = (
            cordx_start,
            cordy_start,
            cordx_start + logo_size0,
            cordy_start + logo_size1,
        )
    region = logo_img.resize((box[2] - box[0], box[3] - box[1]))
    r, g, b, a = region.split()
    img_pip.paste(region, box, a)
    return img_pip


def random_affine_v2(img, **kwargs):
    hori, vert = 0.1 * random.randint(1, 2), 0.1 * random.randint(1, 2)
    ang = random.randint(10, 270)
    trans = T.RandomAffine(
        degrees=ang,
        translate=(hori, vert),
        scale=(0.7, 1.2),
        shear=None,
        resample=Image.BILINEAR,
        fillcolor=0,
    )
    img = trans(img)
    return img


def random_color_v2(img, **kwargs):
    brightness, contrast, saturation, hue = (
        random.random(),
        random.random(),
        random.random(),
        random.uniform(0, 0.5),
    )
    trans = T.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )
    img = trans(img.copy())
    return img
