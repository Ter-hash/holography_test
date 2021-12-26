# -*- coding: utf-8 -*-
# @Time    : 2021/12/26 15:26
# @Author  : Spring
# @FileName: propagation_HIL.py
# @Software: PyCharm
"""
HIL: a 2D wave propagation model that is trained using an SGD-based hardware-in-the-loop training strategy
[Chakravarthula et al. 2020]; once trained, this model is used to generate new holograms using an SGD solver
HIL:基于 SGD-CITL 2D 波传播模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
