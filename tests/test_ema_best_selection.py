"""raw/EMA 双轨 best 选择测试。

DualBestTracker 两轨各自独立 argmax（指标硬编码 mIoU）、覆盖式保存
{ckpt_stem}_best_{raw|ema}.pth；best_ema.pth 的 'model' key 可 strict-load。
"""
import logging
import os
import re
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import DualBestTracker  # noqa: E402


class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Conv2d(4, 1, 1)


def _metrics(miou):
    return {'epoch': -1, 'mIoU': miou, 'ODS': 0.0, 'OIS': 0.0,
            'F1': 0.0, 'Precision': 0.0, 'Recall': 0.0}


def _make_tracker(tmp, with_logger=False):
    logger = None
    log_path = os.path.join(tmp, 'train.log')
    if with_logger:
        logger = logging.getLogger('test_ema_best_selection')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(logging.FileHandler(log_path, mode='w'))
    return DualBestTracker(tmp, 'run_stem', logger=logger), log_path


class TestDualBestSelection(unittest.TestCase):

    def test_tracks_are_independent(self):
        # a) raw 在 ep3 最优、ema 在 ep7 最优，两轨互不污染。
        with tempfile.TemporaryDirectory() as tmp:
            tracker, _ = _make_tracker(tmp)
            model = Mini()
            raw_seq = [0.50, 0.60, 0.70, 0.80, 0.65, 0.55, 0.62, 0.58, 0.61, 0.59]
            ema_seq = [0.52, 0.58, 0.66, 0.72, 0.75, 0.78, 0.81, 0.90, 0.85, 0.83]
            for epoch, (r, e) in enumerate(zip(raw_seq, ema_seq)):
                tracker.maybe_update('raw', _metrics(r), epoch, model, args=None)
                tracker.maybe_update('ema', _metrics(e), epoch, model, args=None)
            self.assertEqual(tracker.state['raw']['epoch'], 3)
            self.assertAlmostEqual(tracker.state['raw']['mIoU'], 0.80)
            self.assertEqual(tracker.state['ema']['epoch'], 7)
            self.assertAlmostEqual(tracker.state['ema']['mIoU'], 0.90)

    def test_overwrite_semantics_single_file_latest_content(self):
        # b) 多次刷新后每类文件只有一个，内容为最新 best 的 epoch/metric。
        with tempfile.TemporaryDirectory() as tmp:
            tracker, _ = _make_tracker(tmp)
            model = Mini()
            for epoch, miou in enumerate([0.5, 0.7, 0.6, 0.9, 0.8]):
                tracker.maybe_update('raw', _metrics(miou), epoch, model, args=None)
            bests = [f for f in os.listdir(tmp) if 'best_raw' in f]
            self.assertEqual(bests, ['run_stem_best_raw.pth'])
            ckpt = torch.load(tracker.best_path('raw'),
                              map_location='cpu', weights_only=False)
            self.assertEqual(ckpt['epoch'], 3)
            self.assertAlmostEqual(ckpt['best_metric'], 0.9)
            self.assertIn('model', ckpt)
            # 旧式全量存档的 key 不应出现在 best 文件中。
            self.assertNotIn('optimizer', ckpt)
            self.assertNotIn('model_ema', ckpt)

    def test_ema_off_only_raw_best(self):
        # c) EMA 关闭（只喂 raw 轨）：只产出 best_raw，无 ema 文件、无异常。
        with tempfile.TemporaryDirectory() as tmp:
            tracker, _ = _make_tracker(tmp)
            model = Mini()
            for epoch, miou in enumerate([0.5, 0.6, 0.55]):
                tracker.maybe_update('raw', _metrics(miou), epoch, model, args=None)
            self.assertTrue(os.path.isfile(tracker.best_path('raw')))
            self.assertFalse(os.path.exists(tracker.best_path('ema')))
            self.assertIsNone(tracker.track_lines('ema'))
            self.assertIn('[raw] max_mIoU -> 0.6', tracker.track_lines('raw'))
            self.assertIn('[raw] max Epoch -> 1', tracker.track_lines('raw'))

    def test_best_ema_model_key_strict_loads(self):
        # d) best_ema.pth 的 'model' key 能 strict-load 进模型实例。
        with tempfile.TemporaryDirectory() as tmp:
            tracker, _ = _make_tracker(tmp)
            model = Mini()
            with torch.no_grad():
                model.conv.weight.fill_(0.123)
            tracker.maybe_update('ema', _metrics(0.77), 5, model, args=None)
            ckpt = torch.load(tracker.best_path('ema'),
                              map_location='cpu', weights_only=False)
            fresh = Mini()
            fresh.load_state_dict(ckpt['model'], strict=True)
            self.assertTrue(torch.allclose(
                fresh.conv.weight, torch.full_like(fresh.conv.weight, 0.123)))

    def test_new_best_log_line_format(self):
        # e) 新 best 日志行格式：new {raw|ema}-best mIoU=X at epoch N -> 路径。
        with tempfile.TemporaryDirectory() as tmp:
            tracker, log_path = _make_tracker(tmp, with_logger=True)
            model = Mini()
            tracker.maybe_update('ema', _metrics(0.654321), 7, model, args=None)
            with open(log_path) as f:
                text = f.read()
            pattern = (r"new ema-best mIoU=[0-9.eE+-]+ at epoch 7 -> "
                       r".*run_stem_best_ema\.pth")
            self.assertRegex(text, pattern)


if __name__ == '__main__':
    unittest.main()
