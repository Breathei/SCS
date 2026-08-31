"""
Micro-benchmark: compare a single forward+backward batch between
Euler (mamba_ssm fused kernel) and Trapezoidal (pure PyTorch loop).
"""
import argparse
import time
import torch
from models import build_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--discretization', default='euler', type=str,
                        choices=['euler', 'trapezoidal_fixed'])
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--load_width', default=512, type=int)
    parser.add_argument('--load_height', default=512, type=int)
    parser.add_argument('--warmup', default=2, type=int)
    parser.add_argument('--iterations', default=5, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)
    args.device = device
    args.BCELoss_ratio = 0.83
    args.DiceLoss_ratio = 0.17
    args.dataset_path = './Dataset/TUT'
    args.batch_size_train = args.batch_size
    args.batch_size_test = 1
    args.lr_scheduler = 'PolyLR'
    args.lr = 5e-4
    args.min_lr = 1e-6
    args.weight_decay = 0.01
    args.sgd = False
    args.output_dir = './checkpoints/weights'
    args.seed = 42
    args.phase = 'train'

    model, criterion = build_model(args)
    model.to(device)
    criterion.to(device)
    model.train()

    x = torch.randn(
        args.batch_size, 3, args.load_height, args.load_width,
        device=device, dtype=torch.float32
    )
    target = torch.ones(
        args.batch_size, 1, args.load_height, args.load_width,
        device=device, dtype=torch.float32
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Warmup.
    for _ in range(args.warmup):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Timed iterations.
    times = []
    for _ in range(args.iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    mean_time = sum(times) / len(times)
    print(f"discretization={args.discretization} batch_size={args.batch_size} "
          f"resolution={args.load_width}x{args.load_height}")
    print(f"  mean fwd+bwd+step over {args.iterations} iters: {mean_time:.3f}s")
    print(f"  per-iteration times: {[f'{t:.3f}' for t in times]}s")


if __name__ == '__main__':
    main()
