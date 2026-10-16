'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch
from torch import nn
import torch.nn.functional as F
from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
from models.MFS import MFS
from models.detail_branch import DetailBranch
from models.degconv import DEGConv

# 4 级 stage 输出通道（SAVSS.forward 固定：conv256to{128,64,32,16} +
# 上采样到 64²/128²/256²/512²）。
_STAGE_CHANNELS = (128, 64, 32, 16)

class Decoder(nn.Module):
    def __init__(self, backbone, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.mfs_channels = getattr(args, 'mfs_channels', 8)
        self.MFS = MFS(self.mfs_channels)
        # 高分辨率细节支路（实验选项，默认关闭）。关闭时不创建任何模块，
        # state_dict 与默认配置完全一致。
        self.use_detail_branch = bool(getattr(args, 'detail_branch', False))
        self.detail_branch = None
        self.detail_delta = None
        self.detail_refine_dw = None
        self.detail_refine_pw = None
        # DEGConv（方向引导边缘门控卷积，实验选项，默认关闭）。开启时插在
        # 每级 stage 输出之后、MFS 之前（4 个独立实例）；关闭时不创建任何
        # 模块，state_dict 与默认配置完全一致。
        self.use_degconv = bool(getattr(args, 'degconv', False))
        self.degconvs = None
        if self.use_degconv:
            self.degconvs = nn.ModuleList(
                [DEGConv(c) for c in _STAGE_CHANNELS])
        if self.use_detail_branch:
            self.detail_branch = DetailBranch()
            # 残差注入 + DWConv3x3 空间修正（最终形态，硬编码）：
            #   delta  = Conv1x1(16→C, 权重偏置全零)(upsample(detail))
            #   fused  = mfs_feat + delta
            #   refine = PWConv1x1(C→C, 零初始化) ∘ DWConv3x3(C→C, groups=C)
            #   fused  = fused + refine(fused)
            # 零初始化保证开启时初始前向与关闭完全一致（等价起点）：
            # detail 有效贡献从零开始，delta/refine_pw 第一步即获得非零梯度，
            # DetailBranch 内部与 refine_dw 经一层零权重隔离，第二 step 起获得梯度。
            self.detail_delta = nn.Conv2d(16, self.mfs_channels, kernel_size=1)
            nn.init.zeros_(self.detail_delta.weight)
            nn.init.zeros_(self.detail_delta.bias)
            self.detail_refine_dw = nn.Conv2d(self.mfs_channels, self.mfs_channels,
                                              kernel_size=3, padding=1,
                                              groups=self.mfs_channels)
            self.detail_refine_pw = nn.Conv2d(self.mfs_channels, self.mfs_channels,
                                              kernel_size=1)
            nn.init.zeros_(self.detail_refine_pw.weight)
            nn.init.zeros_(self.detail_refine_pw.bias)

    def forward(self, samples):
        outs_SAVSS = self.backbone(samples)
        if self.degconvs is not None:
            # DEGConv：每级 stage 输出先经方向引导边缘门控，再进 MFS。
            outs_SAVSS = [d(f) for d, f in zip(self.degconvs, outs_SAVSS)]
        if self.detail_branch is None:
            return self.MFS(outs_SAVSS)
        mfs_feat = self.MFS(outs_SAVSS, return_feat=True)
        detail = self.detail_branch(samples)
        detail = F.interpolate(detail, size=mfs_feat.shape[-2:],
                               mode='bilinear', align_corners=False)
        fused = mfs_feat + self.detail_delta(detail)
        fused = fused + self.detail_refine_pw(self.detail_refine_dw(fused))
        return self.MFS.predict(fused)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true, weight=None):
        # weight=None（默认，α=0 或旧调用方）时走原 BCE 调用，与旧实现逐位一致；
        # 传入时仅 BCE 部分逐像素加权（reduction 仍为 mean，除以 numel，
        # 损失尺度与不加权时相当），Dice 部分不变，比例仍由 args 控制。
        if weight is not None:
            bce = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weight)
        else:
            bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice

def build(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    discretization = getattr(args, 'discretization', 'euler')
    backbone = SAVSS(arch='Crack',
                     out_indices=(0, 1, 2, 3),
                     drop_path_rate=0.2,
                     final_norm=True,
                     convert_syncbn=True,
                     discretization=discretization,
                     lam_nheads=getattr(args, 'lam_nheads', 8),
                     trap_lambda=getattr(args, 'trap_lambda', 0.5),
                     trap_boundary=getattr(args, 'trap_boundary', 'euler'),
                     trap_lambda_per_dir=getattr(args, 'trap_lambda_per_dir', False),
                     scan_routing=getattr(args, 'scan_routing', 'none'),
                     scan_param_order=getattr(args, 'scan_param_order', 'current'),
                     asr_beta=getattr(args, 'asr_beta', 0.1),
                     asr_route_grid=getattr(args, 'asr_route_grid', 1),
                     asr_v2_beta=getattr(args, 'asr_v2_beta', 0.25),
                     num_scans=getattr(args, 'num_scans', 4))
    model = Decoder(backbone, args)
    criterion = bce_dice(args)
    criterion.to(device)

    return model, criterion