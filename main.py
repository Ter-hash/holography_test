"""
Neural holography:

This is the main executive script used for the phase generation using Holonet/UNET or
                                                     optimization using (GS/DPAC/SGD) + camera-in-the-loop (CITL).

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

@article{Peng:2020:NeuralHolography,
author = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
title = {{Neural Holography with Camera-in-the-loop Training}},
journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
year = {2020},
}

-----

$ python main.py --channel=0 --algorithm=HOLONET --root_path=./phases --generator_dir=./pretrained_models
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import configargparse
from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
from propagation_partial import PartialProp
from utils.augmented_image_loader import ImageLoader
from propagation_model import ModelPropagate
from utils.modules import SGD, GS, DPAC, PhysicalProp
from holonet import HoloNet, InitialPhaseUnet, FinalPhaseOnlyUnet, PhaseOnlyUnet
from propagation_ASM import propagation_ASM

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
p.add_argument('--method', type=str, default='SGD', help='Type of algorithm, GS/SGD/DPAC/HOLONET/UNET/?')
p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model, ASM/MODEL/PARTIAL/?')
p.add_argument('--root_path', type=str, default='./phases', help='Directory where optimized phases will be saved.')
p.add_argument('--data_path', type=str, default='./data', help='Directory for the dataset')
p.add_argument('--generator_dir', type=str, default='./pretrained_networks',
               help='Directory for the pretrained holonet/unet network')
p.add_argument('--prop_model_dir', type=str, default='./calibrated_models',
               help='Directory for the CITL-calibrated wave propagation models')
p.add_argument('--citl', type=utils.str2bool, default=False, help='Use of Camera-in-the-loop optimization with SGD')
p.add_argument('--experiment', type=str, default='1', help='Name of experiment')
p.add_argument('--lr', type=float, default=8e-3, help='Learning rate for phase variables (for SGD)')
p.add_argument('--lr_s', type=float, default=2e-3, help='Learning rate for learnable scale (for SGD)')
p.add_argument('--num_iters', type=int, default=500, help='Number of iterations (GS, SGD)')
# for PARTIAL
p.add_argument('--src_type', type=str, default='sLED', help='sLED or LED')

# parse arguments
opt = p.parse_args()
# 实验运行名称
run_id = f'{opt.experiment}_{opt.method}_{opt.prop_model}'  # {algorithm}_{prop_model} format

# 使用 citl
if opt.citl:
    run_id = f'{run_id}_citl'

channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]

print(f'   - optimizing phase with {opt.method}/{opt.prop_model} ... ')
if opt.citl:
    print(f'    - with camera-in-the-loop ...')

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
# 使用部分相干光波传播模型时
if opt.prop_model == "PARTIAL":
    prop_dist = (10 * cm, 10 * cm, 10 * cm)[channel]  # propagation distance from SLM plane to target plane
    wavelength = (634.8 * nm, 510 * nm, 450 * nm)[channel]  # SLED
    if opt.src_type == 'LED':
        wavelength = (633 * nm, 532 * nm, 460 * nm)[channel]  # LED
else:
    prop_dist = (20 * cm, 20 * cm, 20 * cm)[channel]  # propagation distance from SLM plane to target plane
    wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color

feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1080, 1920)  # resolution of SLM
image_res = (1080, 1920)
roi_res = (880, 1600)  # regions of interest (to penalize for SGD)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
# device = torch.device('cuda')  # The gpu you are using
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Options for the algorithm
loss = nn.MSELoss().to(device)  # loss functions to use (try other loss functions!)
s0 = 1.0  # initial scale
root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases

# Tensorboard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

# Hardware setup for CITL
if opt.citl:
    camera_prop = PhysicalProp(channel, laser_arduino=True, roi_res=(roi_res[1], roi_res[0]), slm_settle_time=0.12,
                               range_row=(220, 1000), range_col=(300, 1630),
                               patterns_path=f'./citl/calibration',
                               show_preview=True)
else:
    camera_prop = None

# Simulation model
if opt.prop_model == 'ASM':
    propagator = propagation_ASM  # Ideal model
elif opt.prop_model.upper() == 'MODEL':
    blur = utils.make_kernel_gaussian(0.85, 3)
    propagator = ModelPropagate(distance=prop_dist,  # Parameterized wave propagation model
                                feature_size=feature_size,
                                wavelength=wavelength,
                                blur=blur).to(device)

    # load CITL-calibrated model
    # 将训练好的 CITL-calibrated 模型加载到网络上
    # propagator.load_state_dict(torch.load(f'{opt.prop_model_dir}/{chan_str}.pth', map_location=device))
    propagator.eval()
elif opt.prop_model.upper() == 'PARTIAL':
    propagator = PartialProp(distance=prop_dist, feature_size=feature_size, batch_size=12,
                             wavelength_central=wavelength, num_wvls=15,
                             sample_wavelength_rate=1 * nm,
                             randomly_sampled=True,
                             use_sampling_pool=True,
                             f_col=200 * mm,
                             source_diameter=75 * um,
                             source_amp_sigma=30 * um,
                             src_type=opt.src_type,  # 'sLED' or 'LED'
                             device=device).to(device)
    propagator.eval()
elif opt.prop_model.upper() == 'CNNpropCNN':
    pass

# Select Phase generation method, algorithm
# 全息图优化方法
if opt.method == 'SGD':
    phase_only_algorithm = SGD(prop_dist, wavelength, feature_size, opt.num_iters, roi_res, root_path,
                               opt.prop_model, propagator, loss, opt.lr, opt.lr_s, s0, opt.citl, camera_prop, writer,
                               device)
elif opt.method == 'GS':
    phase_only_algorithm = GS(prop_dist, wavelength, feature_size, opt.num_iters, root_path,
                              opt.prop_model, propagator, writer, device)
elif opt.method == 'DPAC':
    phase_only_algorithm = DPAC(prop_dist, wavelength, feature_size, opt.prop_model, propagator, device)
elif opt.method == 'HOLONET':
    phase_only_algorithm = HoloNet(prop_dist, wavelength, feature_size, initial_phase=InitialPhaseUnet(4, 16),
                                   final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2)).to(device)
    model_path = os.path.join(opt.generator_dir, f'holonet20_{chan_str}.pth')
    image_res = (1072, 1920)
elif opt.method == 'UNET':
    phase_only_algorithm = PhaseOnlyUnet(num_features_init=32).to(device)
    model_path = os.path.join(opt.generator_dir, f'unet20_{chan_str}.pth')
    image_res = (1024, 2048)
elif opt.method == "UNET3D":
    phase_only_algorithm = Unet3D().to(device)
    model_path = os.path.join(opt.generator_dir, f'unet3d_{chan_str}.pth')
    image_res = (1024, 2048)

# 加载训练好的优化网络
if 'NET' in opt.method:
    checkpoint = torch.load(model_path)
    phase_only_algorithm.load_state_dict(checkpoint)
    phase_only_algorithm.eval()

# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
# 加载图像
image_loader = ImageLoader(opt.data_path, channel=channel,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

# Loop over the dataset
# 遍历数据集
for k, target in enumerate(image_loader):
    # get target image
    target_amp, target_res, target_filename = target
    print(target_amp.shape, target_res, target_filename)
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)
    print(target_idx)

    # if you want to separate folders by target_idx or whatever, you can do so here.
    phase_only_algorithm.init_scale = s0 * utils.crop_image(target_amp, roi_res, stacked_complex=False).mean()
    phase_only_algorithm.phase_path = os.path.join(root_path)

    # run algorithm (See algorithm_modules.py and algorithms.py)
    # 最终的相位图
    if opt.method in ['DPAC', 'HOLONET', 'UNET']:
        # direct methods
        _, final_phase = phase_only_algorithm(target_amp)
    else:
        # iterative methods, initial phase: random guess
        init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
        final_phase = phase_only_algorithm(target_amp, init_phase)

    print(final_phase.shape)

    # save the final result somewhere.
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)

    utils.cond_mkdir(root_path)
    cv2.imwrite(os.path.join(root_path, f'{target_idx}.png'), phase_out_8bit)

print(f'    - Done, result: --root_path={root_path}')
