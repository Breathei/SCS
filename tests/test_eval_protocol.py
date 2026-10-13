"""评估口径测试：sigmoid_prob_v1。

训练侧是概率口径（BCEWithLogitsLoss + Dice(sigmoid)），评估必须测量同一
对象：sigmoid(logits) 的 [0,1] float 概率图，固定阈值/扫描区间 [0,1]，
不做逐图 max 归一化。日志 key（mIouU/ODS/OIS/F1/P/R）与旧版逐字一致，
头部新增 eval protocol -> sigmoid_prob_v1。
"""
import logging
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evaluate import (  # noqa: E402
    EVAL_PROTOCOL,
    eval,
    prf_metrics,
    sigmoid_prob,
)


def _make_logger(log_path):
    logger = logging.getLogger('test_eval_protocol')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.FileHandler(log_path, mode='w')
    logger.addHandler(handler)
    return logger


def _synthetic_pairs(n=4, size=96, seed=0):
    """裂缝状 GT（对角线）+ 固定 seed 的噪声概率图。"""
    rng = np.random.RandomState(seed)
    pred_list, gt_list = [], []
    for i in range(n):
        gt = np.zeros((size, size), dtype=np.uint8)
        rr = np.arange(size)
        gt[rr, (rr + i) % size] = 1
        gt[rr, (rr + i + 1) % size] = 1
        prob = np.clip(gt * 0.7 + 0.1 + rng.randn(size, size) * 0.05,
                       0.0, 1.0).astype(np.float32)
        pred_list.append(prob)
        gt_list.append(gt)
    return pred_list, gt_list


class TestEvalProtocol(unittest.TestCase):

    def test_probs_equal_torch_sigmoid(self):
        # a) 数学正确性：进入评估的概率 == torch.sigmoid(logits)，容差 1e-6。
        torch.manual_seed(0)
        logits = torch.randn(2, 64, 64) * 3.0
        probs = sigmoid_prob(logits)
        expected = torch.sigmoid(logits)
        self.assertEqual(probs.dtype, torch.float32)
        self.assertTrue(torch.allclose(probs, expected, atol=1e-6))
        self.assertGreaterEqual(probs.min().item(), 0.0)
        self.assertLessEqual(probs.max().item(), 1.0)

    def test_all_negative_logits_stay_below_half_and_ordered(self):
        # b) 病理用例：全负 logits → 概率全部 <0.5，且像素间相对排序不变。
        logits = torch.linspace(-6.0, -0.2, steps=64 * 48).reshape(48, 64)
        # 打乱空间顺序，避免排序保持是“构造巧合”。
        perm = torch.randperm(logits.numel())
        logits = logits.reshape(-1)[perm].reshape(48, 64)
        probs = sigmoid_prob(logits)
        self.assertLess(probs.max().item(), 0.5)
        self.assertTrue(torch.equal(probs.reshape(-1).argsort(),
                                    logits.reshape(-1).argsort()))

    def test_prf_uses_fixed_half_threshold(self):
        # c) max != 1.0 的概率图必须按固定 0.5 二值化，而非按图最大值归一。
        # 固定 0.5：pred_bin=[[1,0],[0,1]]，tp=2/fp=0/fn=0 → P=R=F1=1；
        # 旧行为（pred/max>0.5 ⇒ pred>0.4）会多出一个 fp，P=2/3。
        pred = [np.array([[0.6, 0.45], [0.2, 0.7]], dtype=np.float32)]
        gt = [np.array([[1, 0], [0, 1]], dtype=np.uint8)]
        p, r, f = prf_metrics(pred, gt)
        self.assertAlmostEqual(p, 1.0, places=7)
        self.assertAlmostEqual(r, 1.0, places=7)
        self.assertAlmostEqual(f, 1.0, places=7)

    def test_eval_is_deterministic(self):
        # d) 同一组 (prob, gt) 输入两次评估，指标完全一致。
        pred_list, gt_list = _synthetic_pairs(n=3)
        with tempfile.TemporaryDirectory() as tmp:
            m1 = eval(_make_logger(os.path.join(tmp, 'e1.log')),
                      pred_list, gt_list, 0)
            m2 = eval(_make_logger(os.path.join(tmp, 'e2.log')),
                      pred_list, gt_list, 0)
        for key in ('mIoU', 'ODS', 'OIS', 'F1', 'Precision', 'Recall'):
            self.assertEqual(m1[key], m2[key], key)

    def test_f1_pr_reported_at_best_threshold(self):
        # F1/P/R 三行必须取 micro-F1 最优阈值行（argmax），而不是退化
        # 的 thresh=0 行（sigmoid 口径下该行恒为 R=1、P=前景占比）。
        # 构造确定性数据：背景概率均匀分布于 [0,0.4]，前景 70% 为 0.8、
        # 30% 为 0.2——最优阈值约 0.5（R=0.7, P=1），thresh=0 行退化。
        from eval.evaluate import cal_prf_metrics
        rng = np.random.RandomState(0)
        pred_list, gt_list = [], []
        for i in range(3):
            gt = np.zeros((64, 64), dtype=np.uint8)
            rr = np.arange(64)
            gt[rr, (rr + i) % 64] = 1
            gt[rr, (rr + i + 1) % 64] = 1
            prob = np.linspace(0.0, 0.4, 64 * 64,
                               dtype=np.float32).reshape(64, 64)
            fg = gt.astype(bool)
            fg_probs = np.where(rng.rand(fg.sum()) < 0.7, 0.8, 0.2)
            prob[fg] = fg_probs.astype(np.float32)
            pred_list.append(prob)
            gt_list.append(gt)
        with tempfile.TemporaryDirectory() as tmp:
            metrics = eval(_make_logger(os.path.join(tmp, 'eval.log')),
                           pred_list, gt_list, 0)
        table = np.array(cal_prf_metrics(pred_list, gt_list))
        best_idx = int(np.argmax(table[:, 3]))
        # thresh=0 行确实是退化的（构造前提），argmax 行不应落在那里。
        self.assertAlmostEqual(float(table[0, 2]), 1.0, places=7)
        self.assertGreater(best_idx, 0)
        self.assertAlmostEqual(metrics['F1'], float(table[best_idx, 3]), places=7)
        self.assertAlmostEqual(metrics['Precision'], float(table[best_idx, 1]), places=7)
        self.assertAlmostEqual(metrics['Recall'], float(table[best_idx, 2]), places=7)
        self.assertLess(metrics['Recall'], 1.0)
        self.assertGreater(metrics['F1'], float(table[0, 3]))

    def test_end_to_end_small_sample_and_log_format(self):
        # e) 端到端小样本：完整 eval 流程，指标在 [0,1]、无 NaN，
        #    日志 key 与旧版一致且含 protocol 头行。
        pred_list, gt_list = _synthetic_pairs(n=4)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, 'eval.log')
            metrics = eval(_make_logger(log_path), pred_list, gt_list, 7)
            with open(log_path) as f:
                log_text = f.read()

        self.assertEqual(metrics['epoch'], 7)
        for key in ('mIoU', 'ODS', 'OIS', 'F1', 'Precision', 'Recall'):
            v = metrics[key]
            self.assertTrue(np.isfinite(v), key)
            self.assertGreaterEqual(v, 0.0, key)
            self.assertLessEqual(v, 1.0, key)

        self.assertIn("eval protocol -> " + EVAL_PROTOCOL, log_text)
        self.assertEqual(EVAL_PROTOCOL, 'sigmoid_prob_v1')
        for key_line in ("mIouU -> ", "ODS -> ", "OIS -> ",
                         "F1 -> ", "P -> ", "R -> ", "eval finish!"):
            self.assertIn(key_line, log_text)
        # protocol 头行必须先于指标行出现。
        self.assertLess(log_text.index("eval protocol ->"),
                        log_text.index("mIouU ->"))


if __name__ == '__main__':
    unittest.main()
