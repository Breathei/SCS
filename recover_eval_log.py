"""
恢复被截断的 eval.log。

原理：每个 epoch 的预测图保存在 ./results/<run>/results_<epoch>/ 下，
eval() 只读取这些 png（不需要模型），因此可以对历史 epoch 重新计算指标，
并按 main.py 原有的日志格式补写回 eval.log。

用法（在项目根目录执行）：
    python recover_eval_log.py \
        --results_root "./results/2026_09_20_13:57:17_Dataset->Crack500" \
        --eval_log "./checkpoints/2026_09_20_13:57:17_Dataset->Crack500/eval.log" \
        --start_epoch 0 --end_epoch 26

注意：补写行的 asctime 时间戳是恢复执行时间，不是原始训练时间。
"""
import argparse

from eval.evaluate import eval
from util.logger import get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', required=True,
                        help='e.g. ./results/2026_09_20_13:57:17_Dataset->Crack500')
    parser.add_argument('--eval_log', required=True,
                        help='要补写的 eval.log 路径（所在目录作为 logger 目录）')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, required=True,
                        help='恢复到的 epoch（含），例如 26 表示恢复 epoch 0..26')
    args = parser.parse_args()

    import os
    log_dir = os.path.dirname(os.path.normpath(args.eval_log))
    log_name = os.path.splitext(os.path.basename(args.eval_log))[0]
    log_eval = get_logger(log_dir, log_name)

    log_eval.info(
        "=================== recovered entries (epochs "
        f"{args.start_epoch}..{args.end_epoch}, regenerated from saved "
        "predictions) ==================="
    )

    max_mIoU = 0.0
    max_epoch = 0
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        save_root = f'{args.results_root}/results_{epoch}'
        print(f'recovering epoch {epoch}: {save_root}', flush=True)
        metrics = eval(log_eval, save_root, epoch)
        if max_mIoU < metrics['mIoU']:
            max_mIoU = metrics['mIoU']
            max_epoch = epoch
        # 与 main.py 训练循环中 eval 之后的三行日志保持一致。
        log_eval.info("evalauting epoch finish -> " + str(epoch))
        log_eval.info('\nmax_mIoU -> ' + str(max_mIoU) + '\nmax Epoch -> ' + str(max_epoch))
        log_eval.info("---------------------------------------------------------------------------------------")
        print(f"epoch {epoch}: mIoU={metrics['mIoU']:.4f} ODS={metrics['ODS']:.4f} "
              f"OIS={metrics['OIS']:.4f} F1={metrics['F1']:.4f}", flush=True)

    log_eval.info(
        f"=================== recovery done (best so far: mIoU={max_mIoU} "
        f"@ epoch {max_epoch}) ==================="
    )
    print(f'done. best mIoU={max_mIoU} @ epoch {max_epoch}')


if __name__ == '__main__':
    main()
