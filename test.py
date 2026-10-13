'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import numpy as np
import torch
import argparse
import os
import cv2
import util.misc as utils
from datasets import create_dataset
from models import build_model
from main import get_args_parser, sanitize_checkpoint_state_dict
from eval.evaluate import sigmoid_prob

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()
args.phase = 'test'

if __name__ == '__main__':
    args.batch_size = 1
    t_all = []
    device = utils.validate_runtime_device(args.device)
    args.device = device
    test_dl = create_dataset(args)
    load_model_file = args.checkpoint_path
    data_size = len(test_dl)
    model, criterion = build_model(args)
    try:
        checkpoint = torch.load(load_model_file, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(load_model_file, map_location=device)
    sanitize_checkpoint_state_dict(
        checkpoint.get("model", checkpoint), model,
        scan_routing=getattr(args, 'scan_routing', 'none')
    )
    model.to(device)
    print("Load Model Successful!")
    suffix = os.path.splitext(os.path.basename(load_model_file))[0]
    save_root = "./results/results_test/" + suffix
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    with torch.no_grad():
        model.eval()
        for batch_idx, (data) in enumerate(test_dl):
            x = data["image"].to(device)
            target = data["label"].to(device=device, dtype=torch.int64)
            out = model(x)

            target = target[0, 0, ...].cpu().numpy()
            # sigmoid_prob_v1：导出口径与训练/评估一致，概率图不做 max 归一化。
            prob = sigmoid_prob(out[0, 0]).cpu().numpy()
            root_name = data["A_paths"][0].split("/")[-1][0:-4]
            target = 255 * (target / np.max(target))
            out = (255 * prob).astype(np.uint8)

            print('----------------------------------------------------------------------------------------------')
            print(os.path.join(save_root, "{}_lab.png".format(root_name)))
            print(os.path.join(save_root, "{}_pre.png".format(root_name)))
            print('----------------------------------------------------------------------------------------------')
            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)

    print("Finished!")
