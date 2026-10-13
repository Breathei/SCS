'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import os
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
from engine import train_one_epoch, ModelEMA
from models import build_model
from datasets import create_dataset
import cv2
from eval.evaluate import eval
from util.logger import get_logger
from tqdm import tqdm
from mmengine.optim.scheduler.lr_scheduler import PolyLR


def sanitize_checkpoint_state_dict(state_dict, model, scan_routing='none', logger=None):
    """Load only compatible checkpoint keys for route-mode transitions.

    Router namespaces are version-aware:
    - legacy router (``router.*``) is used by ``asr`` / ``asr_scan``;
    - v2 router (``router_v2.*``) is used by ``asr_scan_v2``;
    - ``none`` uses neither.

    A checkpoint that completely lacks the target router namespace loads with
    the model's freshly initialized router (reported).  A checkpoint with a
    *partial* router state is rejected.  All non-router keys stay strict.
    """
    state_dict = dict(state_dict)
    ignored = []

    def drop_keys(keys):
        for key in keys:
            ignored.append(key)
            state_dict.pop(key)

    def is_router_v2_key(key):
        return key.startswith('router_v2.') or '.router_v2.' in key

    def is_router_key(key):
        # legacy router namespace, excluding router_v2.*
        return (key.startswith('router.') or '.router.' in key) \
            and not is_router_v2_key(key)

    drop_keys([k for k in list(state_dict) if 'trap_gate_logit' in k])
    drop_keys([k for k in list(state_dict) if k.endswith('last_route_w')])

    target_legacy = scan_routing in ('asr', 'asr_scan')
    target_v2 = scan_routing == 'asr_scan_v2'
    # 目标模式不使用的 router namespace 直接从 source 中剔除并记录。
    if not target_legacy:
        drop_keys([k for k in list(state_dict) if is_router_key(k)])
    if not target_v2:
        drop_keys([k for k in list(state_dict) if is_router_v2_key(k)])

    model_state = model.state_dict()
    missing = sorted(k for k in model_state if k not in state_dict)
    unexpected = sorted(k for k in state_dict if k not in model_state)
    allowed_missing = []

    def check_router_namespace(pred, label):
        # source 完全缺少目标 router → 允许缺失（模型用初始化值）；
        # source 只含部分 router key → 视为损坏的 checkpoint，拒绝加载。
        source_keys = {k for k in state_dict if pred(k)}
        model_keys = {k for k in model_state if pred(k)}
        if not source_keys:
            allowed_missing.extend(k for k in missing if pred(k))
        elif source_keys != model_keys:
            raise RuntimeError(
                f"Checkpoint contains an incomplete {label} router state: "
                f"found={sorted(source_keys)}, expected={sorted(model_keys)}"
            )

    if target_legacy:
        check_router_namespace(is_router_key, 'legacy')
    if target_v2:
        check_router_namespace(is_router_v2_key, 'v2')

    bad_missing = [k for k in missing if k not in allowed_missing]
    if unexpected or bad_missing:
        raise RuntimeError(
            "Checkpoint keys are incompatible: "
            f"missing={bad_missing}, unexpected={unexpected}"
        )

    for k, v in state_dict.items():
        if k in model_state:
            ckpt_shape = tuple(v.shape)
            model_shape = tuple(model_state[k].shape)
            if ckpt_shape != model_shape:
                raise RuntimeError(
                    f"Checkpoint shape mismatch for {k}: "
                    f"checkpoint {ckpt_shape} vs model {model_shape}"
                )

    if ignored:
        msg = "Ignoring checkpoint keys: " + str(ignored)
        if logger is not None:
            logger.info(msg)
        print(msg)
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected_loaded = list(incompatible.unexpected_keys)
    missing_loaded = [k for k in incompatible.missing_keys if k not in allowed_missing]
    if unexpected_loaded or missing_loaded:
        raise RuntimeError(
            "Checkpoint load failed after filtering: "
            f"missing={missing_loaded}, unexpected={unexpected_loaded}"
        )
    if allowed_missing:
        msg = "Initialized missing router keys from model defaults: " + str(allowed_missing)
        if logger is not None:
            logger.info(msg)
        print(msg)
    return ignored, allowed_missing


def get_args_parser():
    parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', add_help=False)

    parser.add_argument('--BCELoss_ratio', default=0.83, type=float,
                        help='Weight ratio for Binary Cross Entropy Loss (0.0-1.0), should sum to 1 with DiceLoss_ratio')
    parser.add_argument('--DiceLoss_ratio', default=0.17, type=float,
                        help='Weight ratio for Dice Loss (0.0-1.0), should sum to 1 with BCELoss_ratio')
    parser.add_argument('--Norm_Type', default='GN', type=str,
                        help='Normalization layer type [GN|BN], GN=GroupNorm')
    parser.add_argument('--dataset_path', default="./Dataset/TUT",
                        help='Root directory path for dataset')
    parser.add_argument('--batch_size_train', type=int, default=1,
                        help='Number of samples per training batch (affects memory usage)')
    parser.add_argument('--batch_size_test', type=int, default=1,
                        help='Number of samples per batch')
    parser.add_argument('--lr_scheduler', type=str, default='PolyLR',
                        help='Learning rate scheduler type [PolyLR|StepLR|CosLR]')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Initial learning rate (base value for schedulers)')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate for PolyLR')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='Weight decay coefficient for regularization')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Total number of training epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Manual epoch number to start training (useful for resuming)')
    parser.add_argument('--lr_drop', default=30, type=int,
                        help='Epoch interval for dropping learning rate in StepLR scheduler')
    parser.add_argument('--sgd', action='store_true',
                        help='Use SGD optimizer instead of default AdamW')
    parser.add_argument('--output_dir', default='./checkpoints/weights',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', default='cuda',
                        help='Computation device [cuda|cpu] for training/inference')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--dataset_mode', type=str, default='crack',
                        help='Dataset mode selector')
    parser.add_argument('--serial_batches', action='store_true',
                        help='Disable random shuffling and use sequential batch sampling if enabled')
    parser.add_argument('--num_threads', default=1, type=int,
                        help='Number of subprocesses for data loading')
    parser.add_argument('--phase', type=str, default='train',
                        help='Runtime phase selector')
    parser.add_argument('--load_width', type=int, default=512,
                        help='Input image width for preprocessing (will be resized)')
    parser.add_argument('--load_height', type=int, default=512,
                        help='Input image height for preprocessing (will be resized)')
    parser.add_argument('--checkpoint_path', default='./checkpoints/weights/checkpoint_TUT/checkpoint_TUT.pth',
                        help='Checkpoint path for standalone test.py inference')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume training from a saved checkpoint path')
    parser.add_argument('--discretization', default='euler', type=str,
                        choices=['euler', 'trapezoid', 'trapezoidal_fixed', 'trapezoidal_data', 'gated_trapezoid'],
                        help='Selective scan discretization: euler (default), '
                             'trapezoid / trapezoidal_fixed (fixed lam=0.5), '
                             'trapezoidal_data (data-dependent lam), '
                             'gated_trapezoid (trainable scalar lambda, Triton fused)')
    parser.add_argument('--lam_nheads', default=8, type=int,
                        help='Number of lam heads for trapezoidal_data (must divide expand*d_model)')
    parser.add_argument('--trap_lambda', default=0.5, type=float,
                        help='Fixed lambda for trapezoidal_fixed; initial lambda for '
                             'gated_trapezoid (per-layer scalar, or per-direction vector '
                             'when --trap_lambda_per_dir is set)')
    parser.add_argument('--trap_boundary', default='euler', type=str,
                        choices=['euler', 'zero_prev'],
                        help='Boundary condition for the first token in gated_trapezoid')
    parser.add_argument('--trap_lambda_per_dir', action='store_true',
                        help='Learn a separate lambda for each of the 4 SASS scan '
                             'directions in gated_trapezoid (default: one scalar per layer)')
    parser.add_argument('--scan_routing', default='none', type=str,
                        choices=['none', 'asr', 'asr_scan', 'asr_scan_v2'],
                        help='Adaptive scan routing: none (default, identical to original '
                             'code), asr (legacy per-image weighted fusion), '
                             'asr_scan (scan-aware residual grid routing), or '
                             'asr_scan_v2 (branch-advantage descriptor routing, opt-in)')
    parser.add_argument('--scan_param_order', default='current', type=str,
                        choices=['current', 'aligned'],
                        help='Selective-scan parameter order: current (legacy behavior) '
                             'or aligned (reorder dt/B/C with each SASS direction)')
    parser.add_argument('--asr_beta', default=0.1, type=float,
                        help='Residual strength for asr_scan fusion (0..1, default 0.1)')
    parser.add_argument('--asr_route_grid', default=1, type=int,
                        choices=[1, 2, 4],
                        help='Coarse spatial grid for asr_scan/asr_scan_v2 routing '
                             '(default: 1)')
    parser.add_argument('--asr_v2_beta', default=0.25, type=float,
                        help='Residual strength for asr_scan_v2 fusion (0..1, '
                             'default 0.25); legacy asr_scan keeps using --asr_beta')
    parser.add_argument('--asr_router_lr_mult', default=1.0, type=float,
                        help='Learning-rate multiplier applied to router_v2 parameters '
                             'only (default 1.0; e.g. 4.0 for v2 experiments)')
    parser.add_argument('--asr_v2_coverage_coef', default=0.01, type=float,
                        help='Coefficient for the asr_scan_v2 anti-collapse coverage '
                             'loss (default 0.01)')
    parser.add_argument('--asr_v2_coverage_cap', default=0.6, type=float,
                        help='Per-layer per-direction average route share cap for the '
                             'asr_scan_v2 coverage loss (default 0.6)')
    parser.add_argument('--asr_v2_balance_warmup_ratio', default=0.1, type=float,
                        help='Fraction of total optimizer steps with zero coverage '
                             'loss, followed by a linear ramp of the same length '
                             '(default 0.1)')
    parser.add_argument('--asr_router_diag', action='store_true',
                        help='Enable periodic detached router diagnostics for '
                             'asr_scan_v2 (written to asr_router_diag.jsonl every '
                             '200 training steps)')
    parser.add_argument('--bal_loss_coef', default=0.01, type=float,
                        help='Coefficient for the ASR balancing loss (default: 0.01)')
    parser.add_argument('--use_ema', action='store_true',
                        help='Enable EMA (exponential moving average, decay=0.999 with '
                             'warmup) of model parameters. EMA weights are only used for '
                             'eval and are saved alongside raw weights in checkpoints '
                             '(default: off; old experiments stay reproducible)')
    parser.add_argument('--boundary_alpha', default=0.0, type=float,
                        help='Boundary weighting strength for BCE: per-pixel weight '
                             'w = 1 + alpha * (dist_to_crack_boundary < tau), applied to '
                             'the BCE term only (e.g. --boundary_alpha 5). alpha=0 '
                             '(default) disables weighting and reproduces the original '
                             'loss exactly (old experiments stay reproducible)')
    parser.add_argument('--boundary_tau', default=3.0, type=float,
                        help='Distance threshold in pixels for the boundary band '
                             '(default: 3). Boundary band masks are cached under '
                             '<dataset_path>/boundary_cache/ keyed by tau and size')
    parser.add_argument('--strong_aug', action='store_true',
                        help='Enable strong albumentations augmentation pipeline on the '
                             'training split only (flips/rot90/ShiftScaleRotate/'
                             'brightness-contrast/gamma/CLAHE/noise/blur; no distortion '
                             'transforms), applied before the resize to load_width/'
                             'load_height. Default off: data flow stays bitwise '
                             'identical to the original pipeline (old experiments '
                             'stay reproducible)')
    parser.add_argument('--skel_loss_coef', default=0.0, type=float,
                        help='Coefficient for the auxiliary SkeletonDistanceLoss '
                             '(two-sided weighting: foreground skeleton + distance-decayed '
                             'background rings; weight map computed per-batch from GT, '
                             'detached). Default 0.0 disables it entirely — the loss is '
                             'neither instantiated nor called, keeping the training path '
                             'bit-identical to the original (old experiments stay '
                             'reproducible)')
    return parser

def main(args):
    checkpoints_path = "./checkpoints"
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    if args.resume:
        run_name = os.path.basename(os.path.dirname(os.path.normpath(args.resume)))
    else:
        run_name = cur_time + '_Dataset->' + dataset_name
    process_folder_path = os.path.join(checkpoints_path, run_name)
    args.phase = 'train'
    if not os.path.exists(process_folder_path):
        os.makedirs(process_folder_path)
    else:
        print("create process folder error!")

    log_train = get_logger(process_folder_path, 'train')
    log_test = get_logger(process_folder_path, 'test')
    log_eval = get_logger(process_folder_path, 'eval')

    log_train.info("args -> " + str(args))
    log_train.info("args: dataset -> " + str(args.dataset_path))
    log_train.info("args: BCELoss_ratio -> " + str(args.BCELoss_ratio))
    log_train.info("args: DiceLoss_ratio -> " + str(args.DiceLoss_ratio))
    print("args: BCELoss_ratio -> " + str(args.BCELoss_ratio))
    print("args: DiceLoss_ratio -> " + str(args.DiceLoss_ratio))

    device = utils.validate_runtime_device(args.device, logger=log_train)
    args.device = device
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)
    args.batch_size = args.batch_size_train

    # 训练开始第一行打印完整配置（读取自实际生效的 args，禁止硬编码）。
    if not 0.0 <= float(getattr(args, 'asr_beta', 0.1)) <= 1.0:
        raise ValueError(f"asr_beta must be in [0, 1], got {args.asr_beta}")
    if int(getattr(args, 'asr_route_grid', 1)) not in (1, 2, 4):
        raise ValueError(
            f"asr_route_grid must be one of (1, 2, 4), got {args.asr_route_grid}"
        )
    if not 0.0 <= float(getattr(args, 'asr_v2_beta', 0.25)) <= 1.0:
        raise ValueError(f"asr_v2_beta must be in [0, 1], got {args.asr_v2_beta}")
    if not 0.0 < float(getattr(args, 'asr_v2_coverage_cap', 0.6)) <= 1.0:
        raise ValueError(
            f"asr_v2_coverage_cap must be in (0, 1], got {args.asr_v2_coverage_cap}"
        )
    if not 0.0 <= float(getattr(args, 'asr_v2_balance_warmup_ratio', 0.1)) <= 1.0:
        raise ValueError(
            "asr_v2_balance_warmup_ratio must be in [0, 1], "
            f"got {args.asr_v2_balance_warmup_ratio}"
        )
    if float(getattr(args, 'asr_router_lr_mult', 1.0)) <= 0.0:
        raise ValueError(
            f"asr_router_lr_mult must be positive, got {args.asr_router_lr_mult}"
        )
    config_str = (
        f"Training config: scan_routing={args.scan_routing}, "
        f"scan_param_order={getattr(args, 'scan_param_order', 'current')}, "
        f"asr_beta={getattr(args, 'asr_beta', 0.1)}, "
        f"asr_route_grid={getattr(args, 'asr_route_grid', 1)}, "
        f"bal_loss_coef={args.bal_loss_coef}, discretization={args.discretization}, "
        f"trap_lambda={args.trap_lambda}, use_ema={args.use_ema}, "
        f"boundary_alpha={args.boundary_alpha}, boundary_tau={args.boundary_tau}, "
        f"strong_aug={args.strong_aug}, skel_loss_coef={args.skel_loss_coef}, "
        f"batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}"
    )
    if args.scan_routing == 'asr_scan_v2':
        config_str += (
            f", asr_v2_beta={getattr(args, 'asr_v2_beta', 0.25)}, "
            f"asr_router_lr_mult={getattr(args, 'asr_router_lr_mult', 1.0)}, "
            f"asr_v2_coverage_coef={getattr(args, 'asr_v2_coverage_coef', 0.01)}, "
            f"asr_v2_coverage_cap={getattr(args, 'asr_v2_coverage_cap', 0.6)}, "
            f"asr_v2_balance_warmup_ratio="
            f"{getattr(args, 'asr_v2_balance_warmup_ratio', 0.1)}, "
            f"asr_router_diag={getattr(args, 'asr_router_diag', False)}"
        )
    print(config_str)
    log_train.info(config_str)

    start_epoch = args.start_epoch
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_ema_state = None
    if args.resume:
        resume_path = args.resume
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        log_train.info("Resume checkpoint -> " + str(resume_path))
        print("Resume checkpoint -> " + str(resume_path))
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=device)
        state_dict = checkpoint.get('model', checkpoint)
        sanitize_checkpoint_state_dict(
            state_dict, model, scan_routing=args.scan_routing, logger=log_train
        )
        resume_optimizer_state = checkpoint.get('optimizer')
        resume_scheduler_state = checkpoint.get('lr_scheduler')
        resume_ema_state = checkpoint.get('model_ema')
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

    # EMA：训练开始时创建参数与 buffers 的 shadow 副本。只在每个
    # optimizer.step() 之后更新，训练前向/反向始终使用原始权重；shadow
    # 仅用于 eval 和 checkpoint 保存。
    model_ema = None
    if args.use_ema:
        model_ema = ModelEMA(model, decay=0.999, warmup_steps=1000)
        if resume_ema_state is not None:
            model_ema.load_state_dict(resume_ema_state)
            log_train.info("EMA state loaded from checkpoint.")
            print("EMA state loaded from checkpoint.")
        log_train.info("EMA enabled: decay=0.999, warmup_steps=1000 (eval/save only)")
        print("EMA enabled: decay=0.999, warmup_steps=1000 (eval/save only)")

    train_dataLoader = create_dataset(args)
    dataset_size = len(train_dataLoader)
    print('The number of training images = %d' % dataset_size)
    log_train.info('The number of training images = %d' % dataset_size)

    # λ 标量参数使用 10 倍学习率，使其在 sigmoid 边界附近也能有效更新。
    # router_v2 参数从 base 组拆出，可使用独立的 LR multiplier（默认 1.0，
    # 此时分组存在与否不影响实际学习率，旧模式参数组完全不变）。
    router_v2_params = [
        p for n, p in model.named_parameters()
        if 'router_v2' in n
    ]
    base_params = [
        p for n, p in model.named_parameters()
        if 'trap_lambda_logit' not in n and 'router_v2' not in n
    ]
    lambda_params = [
        p for n, p in model.named_parameters()
        if 'trap_lambda_logit' in n
    ]
    param_dicts = [
        {"params": base_params, "lr": args.lr},
    ]
    if lambda_params:
        param_dicts.append({"params": lambda_params, "lr": args.lr * 1.0})
        print(f"Using {args.lr * 1.0} learning rate for trap_lambda_logit")
        log_train.info(f"Using {args.lr * 1.0} learning rate for trap_lambda_logit")
    if router_v2_params:
        router_lr = args.lr * float(getattr(args, 'asr_router_lr_mult', 1.0))
        param_dicts.append({"params": router_v2_params, "lr": router_lr})
        print(
            f"Using {router_lr} learning rate for router_v2 "
            f"(mult={getattr(args, 'asr_router_lr_mult', 1.0)})"
        )
        log_train.info(
            f"Using {router_lr} learning rate for router_v2 "
            f"(mult={getattr(args, 'asr_router_lr_mult', 1.0)})"
        )
    if args.sgd:
        print('use SGD!')
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        print('use AdamW!')
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=start_epoch, end=args.epochs)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")

    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
        except (ValueError, RuntimeError) as exc:
            log_train.warning(
                "Optimizer state could not be loaded (likely due to param-group "
                f"changes after adding 10x LR for lambda): {exc}. "
                "Continuing with freshly initialized optimizer state."
            )
            print(
                "WARNING: Optimizer state could not be loaded; using fresh optimizer state."
            )
    if resume_scheduler_state is not None:
        try:
            lr_scheduler.load_state_dict(resume_scheduler_state)
        except (ValueError, RuntimeError) as exc:
            log_train.warning(
                f"LR scheduler state could not be loaded: {exc}. "
                "Continuing with freshly initialized scheduler state."
            )
            print(
                "WARNING: LR scheduler state could not be loaded; using fresh scheduler state."
            )

    output_dir = args.output_dir + '/' + run_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    print("Start processing! ")
    log_train.info("Start processing! ")
    start_time = time.time()
    max_mIoU = 0
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}

    def run_evaluation(epoch, save_root, tag=''):
        """在 test 集上前向一遍（写出 PNG），并用官方阈值扫描计算指标，返回 metrics。

        tag 非空时在日志里标记权重组别（'raw'/'ema'），用于区分两组评估指标；
        tag='' 时输出与旧版逐字节一致（EMA 关闭时走此路径）。
        """
        suffix = f" [{tag}]" if tag else ""
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        pbar = tqdm(total=len(test_dl), desc=f"Initial Loss: Pending")

        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        with torch.no_grad():
            model.eval()
            for batch_idx, (data) in enumerate(test_dl):
                x = data["image"].to(device)
                target = data["label"].to(device=device, dtype=torch.int64)
                out = model(x)
                loss = criterion(out, target.float())
                target = target[0, 0, ...].cpu().numpy()
                out = out[0, 0, ...].cpu().numpy()
                root_name = data["A_paths"][0].split("/")[-1][0:-4]

                target = 255 * (target / np.max(target))
                out = 255 * (out / np.max(out))

                # out[out >= 0.5] = 255
                # out[out < 0.5] = 0

                log_test.info('----------------------------------------------------------------------------------------------')
                log_test.info("loss -> " + str(loss))
                log_test.info(str(os.path.join(save_root, "{}_lab.png".format(root_name))))
                log_test.info(str(os.path.join(save_root, "{}_pre.png".format(root_name))))
                log_test.info('----------------------------------------------------------------------------------------------')
                cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
                cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)
        pbar.close()

        log_test.info("model -> " + str(epoch) + " test finish!" + suffix)
        log_test.info('----------------------------------------------------------------------------------------------')
        print(f"testing epoch finish ->  {epoch}{suffix}")
        print("---------------------------------------------------------------------------------------")

        print(f"evalauting epoch start ->  {epoch}{suffix}")
        if tag:
            log_eval.info(f"===== [{tag}] weights eval, epoch {epoch} =====")
        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(str(key) + ' -> ' + str(value))
        return metrics

    for epoch in range(start_epoch, args.epochs):
        print("---------------------------------------------------------------------------------------")
        print("training epoch start -> ", epoch)
        train_info = train_one_epoch(model, criterion, train_dataLoader, optimizer, epoch, args, log_train, ema=model_ema)
        lr_scheduler.step()
        if args.output_dir:
            # Checkpoint filename stem: date + time + dataset_name (without the
            # legacy "Dataset->" separator that is kept in the run folder name).
            ckpt_stem = run_name.replace('Dataset->', '')
            checkpoint_paths = [output_dir / f'{ckpt_stem}.pth']
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'{ckpt_stem}_epoch{epoch}.pth')
            for checkpoint_path in checkpoint_paths:
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                # 开 EMA 时 checkpoint 同时保存原始权重（'model'）和 shadow 权重
                # （'model_ema'）两个 key；不开时不加该 key，保持旧格式。
                if model_ema is not None:
                    save_dict['model_ema'] = model_ema.state_dict()
                utils.save_on_master(save_dict, checkpoint_path)
        print("training epoch finish -> ", epoch)

        # Log learnable λ statistics for gated_trapezoid layers.
        if args.discretization == 'gated_trapezoid':
            lambda_items = [
                (name, torch.sigmoid(m.trap_lambda_logit))
                for name, m in model.named_modules()
                if hasattr(m, 'trap_lambda_logit')
            ]
            if lambda_items:
                lines = [f"Epoch {epoch} gated params (per layer):"]
                for name, lam_tensor in lambda_items:
                    if lam_tensor.dim() == 0:
                        # scalar per-layer λ
                        lines.append(f"  {name}: lambda={lam_tensor.item():.4f}")
                    else:
                        # per-direction λ, shape (4,)
                        vals = [f"{v:.4f}" for v in lam_tensor.tolist()]
                        lines.append(
                            f"  {name}: lambda_dir=[{', '.join(vals)}]"
                        )
                msg = "\n".join(lines)
                log_train.info("\n" + msg)
                print(msg)

        # Log ASR balancing loss and per-layer 4-way average route weights.
        avg_bal_loss = train_info.get('bal_loss', 0.0)
        print(f"Epoch {epoch} L_bal: {avg_bal_loss:.6f}")
        log_train.info(f"Epoch {epoch} L_bal: {avg_bal_loss:.6f}")

        if 'coverage_loss' in train_info:
            cover_msg = (
                f"Epoch {epoch} L_cover: {train_info['coverage_loss']:.6f} "
                f"(coef_eff={train_info['coverage_coef_eff']:.6f})"
            )
            print(cover_msg)
            log_train.info(cover_msg)

        route_w_avgs = train_info.get('route_w_avgs', {})
        if route_w_avgs:
            lines = [f"Epoch {epoch} ASR route weights (per layer):"]
            for name, vals in route_w_avgs.items():
                lines.append(
                    f"{name}: route_w=[{vals[0]:.4f},{vals[1]:.4f},"
                    f"{vals[2]:.4f},{vals[3]:.4f}]"
                )
            msg = "\n".join(lines)
            print(msg)
            log_train.info("\n" + msg)

        print("---------------------------------------------------------------------------------------")

        print("testing epoch start -> ", epoch)
        results_path = cur_time + '_Dataset->' + dataset_name
        save_root = f'./results/{results_path}/results_' + str(epoch)
        # 开 EMA 时每个 epoch 评估两组权重：raw（训练权重，路径与旧行为一致）
        # 和 ema（shadow 权重；评估完立即切回训练权重继续训练）。
        metrics = run_evaluation(epoch, save_root, tag='raw' if model_ema is not None else '')
        if model_ema is not None:
            backup = model_ema.apply_to(model)
            metrics_ema = run_evaluation(epoch, save_root + '_ema', tag='ema')
            model_ema.restore(model, backup)
            ema_summary = (
                f"epoch {epoch} eval summary -> "
                f"raw: mIoU={metrics['mIoU']:.4f} ODS={metrics['ODS']:.4f} "
                f"OIS={metrics['OIS']:.4f} F1={metrics['F1']:.4f} | "
                f"ema: mIoU={metrics_ema['mIoU']:.4f} ODS={metrics_ema['ODS']:.4f} "
                f"OIS={metrics_ema['OIS']:.4f} F1={metrics_ema['F1']:.4f}"
            )
            print(ema_summary)
            log_eval.info(ema_summary)
        if(max_mIoU < metrics['mIoU']):
            max_Metrics = metrics
            max_mIoU = metrics['mIoU']
            ckpt_stem = run_name.replace('Dataset->', '')
            checkpoint_paths = [output_dir / f'{ckpt_stem}_best.pth']
            for checkpoint_path in checkpoint_paths:
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if model_ema is not None:
                    save_dict['model_ema'] = model_ema.state_dict()
                utils.save_on_master(save_dict, checkpoint_path)
            log_train.info("\nupdate and save best model -> " + str(epoch))
            print("\nupdate and save best model -> ", epoch)

        print("evalauting epoch finish -> ", epoch)
        print('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        print("---------------------------------------------------------------------------------------")

        log_eval.info("evalauting epoch finish -> " + str(epoch))
        log_eval.info('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        log_eval.info("---------------------------------------------------------------------------------------")

    for key, value in max_Metrics.items():
        log_eval.info(str(key) + ' -> ' + str(value))
    log_eval.info('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Process time {}'.format(total_time_str))
    log_train.info('Process time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
