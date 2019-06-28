#-*- coding:utf-8 _*-
"""
@author:zjw
@file: mobilenetv3.py
@time: 2019/06/28
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM_DEFAULT = 0.1
BN_EPS_DEFAULT = 1e-5


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def round_channels(channels, depth_multiplier=1.0, depth_divisor=8, min_depth=None):
    """Round number of filters based on depth multiplier."""
    if not depth_multiplier:
        return channels

    channels *= depth_multiplier
    min_depth = min_depth or depth_divisor
    new_channels = max(
        int(channels + depth_divisor / 2) // depth_divisor * depth_divisor,
        min_depth)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += depth_divisor
    return new_channels


def swish(x):
    return x * torch.sigmoid(x)


def hard_swish(x, inplace=True):

    return x * hard_sigmoid(x)

###  relu/6 when convert onnx to ncnn it will cause fatal error
def hard_sigmoid(x, inplace=True):
    relu = F.relu6(x+3.0)
    return (1.0/6)*relu


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_fn=hard_swish,
                 bn_momentum=BN_MOMENTUM_DEFAULT, bn_eps=BN_EPS_DEFAULT):
        super(ConvBnAct, self).__init__()
        assert stride in [1, 2]
        self.act_fn = act_fn
        padding = get_padding(kernel_size, stride)

        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_fn=F.relu,
                 se_ratio=0., se_gate_fn=hard_sigmoid,
                 batchnorm2d=nn.BatchNorm2d, bn_momentum=BN_MOMENTUM_DEFAULT, bn_eps=BN_EPS_DEFAULT):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.act_fn = act_fn
        dw_padding = kernel_size // 2

        self.conv_dw = nn.Conv2d(
            in_chs, in_chs, kernel_size,
            stride=stride, padding=dw_padding, groups=in_chs, bias=False)
        self.bn1 = batchnorm2d(in_chs, momentum=bn_momentum, eps=bn_eps)

        if self.has_se:
            self.se = SqueezeExcite(
                in_chs, reduce_chs=max(1, int(in_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        self.conv_pw = nn.Conv2d(in_chs, out_chs, 1, padding=0, bias=False)
        self.bn2 = batchnorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)

        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu, gate_fn=hard_sigmoid):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduced_chs = reduce_chs or in_chs
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool bad for NVIDIA AMP performance
        # tensor.view + mean bad for ONNX export (produces mess of gather ops that break TensorRT)
        #x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        # x = x * self.gate_fn(x_se)
        return x * self.gate_fn(x_se)


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, kernel_size, dilation=1,
                 stride=1, act_fn=F.relu, exp_ratio=1.0, noskip=False,
                 se_ratio=0., se_reduce_mid=True, se_gate_fn=hard_sigmoid,
                 batchnorm2d=nn.BatchNorm2d, bn_momentum=BN_MOMENTUM_DEFAULT, bn_eps=BN_EPS_DEFAULT):
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        dw_padding = kernel_size // 2

        # Point-wise expansion
        self.conv_pw = nn.Conv2d(in_chs, mid_chs, 1, padding=0, bias=False)
        self.bn1 = batchnorm2d(mid_chs, momentum=bn_momentum, eps=bn_eps)

        # Depth-wise convolution
        self.conv_dw = nn.Conv2d(
            mid_chs, mid_chs, kernel_size, padding=dw_padding*dilation, stride=stride, dilation=dilation, groups=mid_chs, bias=False)
        self.bn2 = batchnorm2d(mid_chs, momentum=bn_momentum, eps=bn_eps)

        # Squeeze-and-excitation
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(
                mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        # Point-wise linear projection
        self.conv_pwl = nn.Conv2d(mid_chs, out_chs, 1, padding=0, bias=False)
        self.bn3 = batchnorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, input):

        # Point-wise expansion
        x = self.conv_pw(input)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x, inplace=True)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            return input+x
        else:
            return x


class MobileNetV3(nn.Module):
    """
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_are_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_are', 'ir_r1_k3_s1_e3_c24_are'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_are'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    """
    def __init__(self, n_class=1000, width_mult=1., bn_momentum=BN_MOMENTUM_DEFAULT):
        super(MobileNetV3, self).__init__()

        max_grid_rate = 4  ###used for 3x3 conv

        base_channel = 16

        # repeate time, kernel, stride, expansion ratio, channel out, if relu, se_ratio(0 reprents no se)
        self.ir_block_setting = [
            [[1, 3, 2, 4, 24, 1, 0],[1, 3, 1, 3, 24, 1, 0]],
            [[3, 5, 2, 3, 40, 1, 0.25]],
            [[1, 3, 2, 6, 80, 0, 0], [1, 3, 1, 2.5, 80, 0, 0], [2, 3, 1, 2.3, 80, 0, 0]],
            [[2, 3, 1, 6, 112, 0, 0.25]],
            [[3, 5, 2, 6, 160, 0, 0.25]]
        ]
        out_ch = round_channels(base_channel, width_mult)
        self.conv_stem = nn.Conv2d(3, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=bn_momentum, eps=BN_EPS_DEFAULT)
        self.hswish = hard_swish

        stages = []
        in_ch = out_ch
        stage0 = nn.Sequential(DepthwiseSeparableConv(in_ch, in_ch, kernel_size=3, stride=1, act_fn=F.relu, bn_momentum=bn_momentum))
        stages.append(stage0)

        self.fea_chs = []

        for stage_setting in self.ir_block_setting:
            blocks = []
            for block_set in stage_setting:
                r,k,s,e,c,relu,se = block_set
                if s==2:
                    self.fea_chs.append(in_ch)
                if relu==1:
                    act_fn = F.relu
                else:
                    act_fn = hard_swish
                for i in range(r):
                    s = s if i==0 else 1
                    out_ch = round_channels(c, width_mult)
                    blocks.append(InvertedResidual(in_ch, out_ch, kernel_size=k, stride=s, act_fn=act_fn, exp_ratio=e, se_ratio=se, bn_momentum=bn_momentum))
                    in_ch=out_ch
            stage = nn.Sequential(*blocks)
            stages.append(stage)

        out_ch = round_channels(960, width_mult)
        last_stage = nn.Sequential(ConvBnAct(in_ch, out_ch, kernel_size=1, stride=1, act_fn=hard_swish, bn_momentum=bn_momentum))
        in_ch = out_ch
        self.fea_chs.append(in_ch)
        stages.append(last_stage)

        self.blocks = nn.Sequential(*stages)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(in_ch, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.classifier = nn.Linear(1280, n_class)

        self.c1_ch, self.c2_ch, self.c3_ch, self.c4_ch, self.c5_ch = self.fea_chs

    def forward(self, x):
        x = self.hswish(self.bn1(self.conv_stem(x)))
        c5= self.blocks(x)

        pool_fea = self.avg_pool(c5)
        fea = self.hswish(self.conv_head(pool_fea))
        fea=fea.view(fea.size(0), -1)
        res = self.classifier(fea)
        return res

