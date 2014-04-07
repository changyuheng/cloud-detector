# -*- coding: utf-8 -*-

import numpy


class HYTA():
    TS = 0.03

    def __init__(self, src, options=None, mask=None):
        self.src = src.astype(numpy.uint8)
        self.options = options
        self.mask = mask

    def normalize_b_r_ratio(self):
        red_channel = self.src[..., 0].copy()
        blue_channel = self.src[..., 2]
        red_channel[red_channel == 0] = 1
        if self.mask:
            red_channel = red_channel[self.mask[..., 0] > 0]
            blue_channel = blue_channel[self.mask[..., 2] > 0]

        lambda_array = blue_channel / red_channel
        lambda_n_array = (lambda_array - 1) / (lambda_array + 1)

        return lambda_n_array.std()

    def run(self):
        std = self.normalize_b_r_ratio()
        print(std)

        return self.src
