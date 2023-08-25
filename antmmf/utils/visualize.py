# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, List, Optional, Tuple
import numpy
import torch
import torchvision


class VisualizationDataRecord:
    r"""
    A data record for storing attribution relevant information
    """
    __slots__ = [
        "word_attributions",
        "pred_prob",
        "pred_class",
        "true_class",
        "attr_class",
        "attr_score",
        "raw_input",
        "convergence_score",
    ]

    def __init__(
        self,
        word_attributions,
        pred_prob,
        pred_class,
        true_class,
        attr_class,
        attr_score,
        raw_input,
        convergence_score,
    ):
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.true_class = true_class
        self.attr_class = attr_class
        self.attr_score = attr_score
        self.raw_input = raw_input
        self.convergence_score = convergence_score


def visualize_images(
    images: List[Any], size: Optional[Tuple[int, int]] = (224, 224), *args, **kwargs
):
    """Visualize a set of images using torchvision's make grid function. Expects
    PIL images which it will convert to tensor and optionally resize them. If resize is
    not passed, it will only accept a list with single image

    Args:
        images (List[Any]): List of images to be visualized
        size (Optional[Tuple[int, int]], optional): Size to which Images can be resized.
            If not passed, the function will only accept list with single image.
            Defaults to (224, 224).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Visualization tools require matplotlib. "
            + "Install using pip install matplotlib."
        )
        raise

    transform_list = []

    assert (
        size is not None or len(images) == 1
    ), "If size is not passed, only one image can be visualized"

    if size is not None:
        transform_list.append(torchvision.transforms.Resize(size=size))

    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)

    img_tensors = torch.stack([transform(image) for image in images])
    grid = torchvision.utils.make_grid(img_tensors, *args, **kwargs)

    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))


def visualize_text_importance(visual_record, pad_token, title):
    import matplotlib.pyplot as plt
    import seaborn as sns

    assert isinstance(visual_record, list)

    # for truncation
    max_pad_pos = -1

    scores = []
    strings = []
    y_axis_labels = []
    for vr in visual_record:
        scores.append(vr.word_attributions)
        strings.append(vr.raw_input)

        for idx, st in enumerate(vr.raw_input):
            if st == pad_token:
                max_pad_pos = max(max_pad_pos, idx)
                break
        y_axis_labels.append(
            "target {} pred {} ({:,.2f})".format(
                vr.true_class, vr.pred_class, vr.pred_prob.round(2)
            )
        )
    strings = numpy.asarray(strings)[:, :max_pad_pos]
    scores = numpy.array(scores)[:, :max_pad_pos]

    # replace [PAD] with ''
    for r in range(strings.shape[0]):
        for c in range(strings.shape[1]):
            if strings[r][c] == pad_token:
                strings[r][c] = ""

    fig, ax = plt.subplots()
    sns.heatmap(
        scores, annot=strings, fmt="", cmap="RdYlGn", ax=ax, yticklabels=y_axis_labels
    ).set_title(title)
    plt.show()


def visualize_image_importance(visual_record, title):
    import matplotlib.pyplot as plt
    import seaborn as sns

    assert isinstance(visual_record, list)

    scores = []
    y_axis_labels = []
    for vr in visual_record:
        scores.append(vr.word_attributions)
        y_axis_labels.append(
            "target {} pred {} ({:,.2f})".format(
                vr.true_class, vr.pred_class, vr.pred_prob.round(2)
            )
        )
    scores = numpy.array(scores)

    fig, ax = plt.subplots()
    sns.heatmap(
        scores, fmt="", cmap="RdYlGn", ax=ax, yticklabels=y_axis_labels
    ).set_title(title)
    plt.show()
