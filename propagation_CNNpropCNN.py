# -*- coding: utf-8 -*-
# @Time    : 2021/12/26 14:54
# @Author  : Spring
# @FileName: propagation_CNNpropCNN.py
# @Software: PyCharm
"""
propCNN: a wave propagation model that uses only a single CNN operating on the complex-valued field
after ASM-based propagation to the target plane(s)

CNNprop: a wave propagation model that uses only a single CNN operating on the complex-valued field
at the SLM plane before propagation to the target plane(s)

CNNpropCNN: a wave propagation model using CNNs operating on the complex-valued field at the SLM plane
before ASM propagation and also directly after propagation to the target planes
"""

import math

import cv2
import torch
import numpy as np
import utils.utils as utils
import torch.fft

from propagation_ASM import propagation_ASM


class CNNprop:
    """
    CNN_slm
    input:全息图
    input: two channels with the real and imaginary values of the field on the SLM
    outputs: the real and imaginary components of the adjusted field at the SLM plane
    output:优化过的全息图
    """

    def __init__(self, input):
        self.input = input

    def optm_input(self):
        print(self.input.shape)


class propCNN:
    """
    CNN_target
    input:经过ASM传播过来的多个目标平面
    输出：经过CNN调整后的多个目标平面
    """
    pass


class CNNpropCNN:
    """
    1.用CNNslm优化
    2.将1中的结果经过ASM处理
    3.将2中的多个结果由CNNtarget优化
    4.返回不同平面的结果
    """

    def __init__(self, u_in):
        self.u_in = u_in

    def to_ASM(self):
        """
        propagation_ASM(self.u_in, feature_size, wavelength, z, linear_conv=True,
        padtype='zero', return_H=False, precomped_H=None,
        return_H_exp=False, precomped_H_exp=None,
        dtype=torch.float32):
        """
        slm_res = (1080, 1920)
        init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to("cuda")
        return propagation_ASM(torch.empty(*init_phase.shape, dtype=torch.complex64), (6.4 * 1e-6, 6.4 * 1e-6),
                               634.8 * 1e-9, 10 * 1e-2, )


if __name__ == "__main__":
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    phases_name = "./phases/1_SGD_ASM/blue/1.png"
    img = cv2.imread(phases_name)
    cnn_prop_cnn = CNNpropCNN(img)
    print(cnn_prop_cnn.to_ASM().shape)
