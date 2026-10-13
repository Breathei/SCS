"""
ASR 验收测试 Gate D：回归测试。

scan_routing='none' 模式下，用官方权重在 DeepCrack 测试划分上推理，
比较 mIoU 与未改动代码的基线（差异 < 0.0001）。

先运行 --save_baseline 保存基线，之后代码改动后运行 --baseline_json 做对比。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import create_dataset
from eval.evaluate import cal_mIoU_metrics
from main import get_args_parser, sanitize_checkpoint_state_dict
from models import build_model


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASELINE_JSON = os.path.join(REPO_ROOT, 'baseline_deepcrack.json')


def main():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument('--save_baseline', type=str, default=None,
                        help='Path to save baseline mIoU JSON')
    parser.add_argument('--baseline_json', type=str, default=DEFAULT_BASELINE_JSON,
                        help='Path to baseline mIoU JSON to compare against')
    args = parser.parse_args()

    if args.save_baseline:
        baseline_json = None
    else:
        baseline_json = args.baseline_json
        if not baseline_json:
            parser.error('--baseline_json must be non-empty unless --save_baseline is used')
        if not os.path.isfile(baseline_json):
            parser.error(
                f'Baseline JSON not found: {baseline_json}. '
                'Pass --baseline_json or use --save_baseline to create one.'
            )

    args.phase = 'test'
    args.batch_size = 1
    args.scan_routing = 'none'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, _ = build_model(args)
    ckpt = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    sanitize_checkpoint_state_dict(
        ckpt.get('model', ckpt), model, scan_routing=args.scan_routing
    )
    model.to(args.device).eval()

    test_dl = create_dataset(args)
    preds, gts = [], []
    with torch.no_grad():
        for data in test_dl:
            x = data['image'].to(args.device)
            target = data['label'].to(args.device, dtype=torch.int64)
            out = model(x)

            target = target[0, 0].cpu().numpy()
            out = out[0, 0].cpu().numpy()

            if target.max() > 0:
                target = 255 * (target / target.max())
            if out.max() > 0:
                out = 255 * (out / out.max())

            preds.append(out)
            gts.append(target)

    miou = cal_mIoU_metrics(preds, gts)
    print(f"mIoU = {miou:.6f}")

    if args.save_baseline:
        with open(args.save_baseline, 'w') as f:
            json.dump({'miou': miou}, f)
        print(f"Saved baseline to {args.save_baseline}")
    else:
        with open(baseline_json) as f:
            baseline = json.load(f)['miou']
        diff = abs(miou - baseline)
        print(f"Baseline mIoU = {baseline:.6f}, diff = {diff:.6e}")
        assert diff < 1e-4, f"Regression too large: {diff}"
        print("PASS")


if __name__ == '__main__':
    main()
