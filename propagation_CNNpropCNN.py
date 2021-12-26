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

import configargparse
import cv2
import torch
import numpy as np
import utils.utils as utils
import torch.fft

from propagation_ASM import propagation_ASM
from utils.augmented_image_loader import ImageLoader


class CNNprop:
    """
    CNN_slm
    input:SLM显示的相位信息，来自于相机捕获 2*M*N
    input: two channels with the real and imaginary values of the field on the SLM
    outputs: the real and imaginary components of the adjusted field at the SLM plane
    output:优化过的全息图 2*M*N
    """

    def __init__(self, u_in):
        self.u_in = u_in

    def optim_slm(self):
        """
        优化输入的相位图
        """
        print(f"1.优化 SLM 完成 {self.u_in.shape}")
        return self.u_in


class propCNN:
    """
    CNN_target
    input:经过ASM传播过来的多个目标平面 2*M*N
    output：经过CNN调整后的多个目标平面 2*M*N
    """

    def __init__(self, u_in):
        self.u_in = u_in

    def optim_target(self):
        print(f"3.优化 target 完成 {self.u_in.shape}")
        return self.u_in


class CNNpropCNN:
    """
    1.用CNNslm优化
    2.将1中的结果经过ASM处理
    3.将2中的多个结果由CNNtarget优化
    4.返回不同平面的结果
    """

    def __init__(self, u_in):
        self.u_in = u_in

    def process(self):
        cnn_prop = CNNprop(res)
        optim_slm = cnn_prop.optim_slm()
        print("2.ASM 转换过程(1->j)")
        prop_cnn = propCNN(res)
        prop_cnn = prop_cnn.optim_target()


class PASM:
    def __init__(self):
        pass


def convert_phase_input(target_amp):
    """
    将SLM_phase 转换为 2*M*N
    the real and imaginary values of the field on the SLM
    为 CNN 提供输入
    """
    target_amp = target_amp.to(device)
    # 由ASM变换
    res = propagation_ASM(target_amp, feature_size, wavelength, prop_dist, )
    # res = target_amp
    res = res[0, 0, :, :]
    # print(res.shape)
    # 拆分实部和虚部，构造CNN的输入
    res = torch.stack((res.real, res.imag))
    # print(res.shape)
    return res


if __name__ == "__main__":
    # Command line argument processing
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
    p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
    p.add_argument('--data_path', type=str, default='./data', help='Directory for the dataset')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # parse arguments
    opt = p.parse_args()
    channel = opt.channel
    # 超参数
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
    wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color
    prop_dist = (20 * cm, 20 * cm, 20 * cm)[channel]  # propagation distance from SLM plane to target plane
    image_res = (1080, 1920)
    # regions of interest (to penalize for SGD)
    roi_res = (880, 1600)
    # 加载数据集
    image_loader = ImageLoader("./data", channel=0, image_res=image_res, homography_res=roi_res,
                               crop_to_homography=True,
                               shuffle=False, vertical_flips=False, horizontal_flips=False)

    for k, target in enumerate(image_loader):
        print(f"图片 {k}")
        target_amp, target_res, target_filename = target
        print(target_amp.shape, target_res, target_filename)
        res = convert_phase_input(target_amp)
        # CNNpropCNN 模型
        cnn_prop = CNNpropCNN(res)
        cnn_prop.process()
        print("\n")
