# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

import torchvision.transforms as pth_transforms

class ImgPilColorDistortion(object):
    """
    Apply Random color distortions to the input image.
    There are multiple different ways of applying these distortions.
    This implementation follows SimCLR - https://arxiv.org/abs/2002.05709
    It randomly distorts the hue, saturation, brightness of an image and can
    randomly convert the image to grayscale.
    """

    def __init__(self, strength):
        """
        Args:
            strength (float): A number used to quantify the strength of the
                              color distortion.
        """
        self.strength = strength
        self.color_jitter = pth_transforms.ColorJitter(
            0.8 * self.strength,
            0.8 * self.strength,
            0.8 * self.strength,
            0.2 * self.strength,
        )
        self.rnd_color_jitter = pth_transforms.RandomApply([self.color_jitter], p=0.8)
        self.rnd_gray = pth_transforms.RandomGrayscale(p=0.2)
        self.transforms = pth_transforms.Compose([self.rnd_color_jitter, self.rnd_gray])

    def __call__(self, image):
        return self.transforms(image)

