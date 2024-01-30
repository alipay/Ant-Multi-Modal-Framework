# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import numpy as np


def get_palette():
    # fmt: off
    basic_colors = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
        ],
        dtype=np.float,
    ).reshape((-1, 3))
    # fmt: on

    # generate palette by some equations.
    other_colors = np.concatenate(
        [np.ones((1, 3), dtype=np.float) * k / 7 for k in range(1, 8)]
        + [
            np.roll([k / 6.0, 0.0, 0.0], i).astype(np.float).reshape((1, 3))
            for i in range(3)
            for k in range(1, 7)
        ]
        + [
            np.array([[b, g, r]], dtype=np.float)
            for r in np.arange(3).astype(np.float) / 2.0
            for g in np.arange(4).astype(np.float) / 3
            for b in np.arange(4).astype(np.float) / 3
        ],
        axis=0,
    )

    palette = np.concatenate([basic_colors, other_colors], axis=0)
    palette = np.unique(palette, axis=0)

    return palette
