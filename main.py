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
                        choices=['none', 'asr'],
                        help='Adaptive scan routing: none (default, identical to original '
                             'code) or asr (per-image weighted fusion of 4 SASS directions)')
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
    config_str = (
        f"Training config: scan_routing={args.scan_routing}, "
        f"bal_loss_coef={args.bal_loss_coef}, discretization={args.discretization}, "
        f"trap_lambda={args.trap_lambda}, use_ema={args.use_ema}, "
        f"boundary_alpha={args.boundary_alpha}, boundary_tau={args.boundary_tau}, "
        f"strong_aug={args.strong_aug}, skel_loss_coef={args.skel_loss_coef}, "
        f"batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}"
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
        # Backward compatibility: older checkpoints may contain the removed
        # trap_gate_logit parameter. Strip it before loading.
        stale_gate_keys = [k for k in state_dict if 'trap_gate_logit' in k]
        if stale_gate_keys:
            for k in stale_gate_keys:
                state_dict.pop(k)
            log_train.info(
                "Ignoring stale trap_gate_logit parameters from checkpoint: "
                + str(stale_gate_keys)
            )
            print(
                "Ignoring stale trap_gate_logit parameters from checkpoint: "
                + str(stale_gate_keys)
            )
        # ASR 监控 buffer last_route_w 初始注册为 None（不计入 state_dict），
        # 训练中变成张量后会被存进 checkpoint；加载到 None buffer 会报
        # unexpected key。它只是日志用的 detached 副本，首个 forward 即被
        # 覆盖，直接剔除。
        route_w_buf_keys = [k for k in state_dict if k.endswith('last_route_w')]
        if route_w_buf_keys:
            for k in route_w_buf_keys:
                state_dict.pop(k)
            log_train.info(
                "Ignoring last_route_w monitoring buffers from checkpoint: "
                + str(route_w_buf_keys)
            )
            print(
                "Ignoring last_route_w monitoring buffers from checkpoint: "
                + str(route_w_buf_keys)
            )
        # Explicit shape check for trap_lambda_logit to give a clear error when
        # switching between scalar and per-direction checkpoints.
        model_state = model.state_dict()
        for k in state_dict:
            if 'trap_lambda_logit' in k and k in model_state:
                ckpt_shape = tuple(state_dict[k].shape)
                model_shape = tuple(model_state[k].shape)
                if ckpt_shape != model_shape:
                    raise RuntimeError(
                        f"Checkpoint shape mismatch for {k}: "
                        f"checkpoint {ckpt_shape} vs model {model_shape}. "
                        f"This usually happens when resuming a checkpoint saved with "
                        f"scalar lambda into a model with --trap_lambda_per_dir (or vice versa). "
                        f"Please ensure --trap_lambda_per_dir matches the checkpoint."
                    )
        model.load_state_dict(state_dict, strict=True)
        resume_optimizer_state = checkpoint.get('optimizer')
        resume_scheduler_state = checkpoint.get('lr_scheduler')
        resume_ema_state = checkpoint.get('model_ema')
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

    # EMA：训练开始时创建参数 shadow 副本（含 GN 的 weight/bias；GN 无 running
    # buffer，无需特殊处理）。只在每个 optimizer.step() 之后更新，训练前向/反向
    # 始终使用原始权重；shadow 仅用于 eval 和 checkpoint 保存。
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
    base_params = [
        p for n, p in model.named_parameters()
        if 'trap_lambda_logit' not in n
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
