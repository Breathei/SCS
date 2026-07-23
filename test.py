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
from main import get_args_parser

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()
args.phase = 'test'
args.dataset_path = '../data/TUT'

if __name__ == '__main__':
    args.batch_size = 1
    t_all = []
    device = utils.validate_runtime_device(args.device)
    args.device = device
    test_dl = create_dataset(args)
    load_model_file = "./checkpoints/weights/checkpoint_TUT/checkpoint_TUT.pth"
    data_size = len(test_dl)
    model, criterion = build_model(args)
    try:
        state_dict = torch.load(load_model_file, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(load_model_file, map_location=device)
    model.load_state_dict(state_dict.get("model", state_dict))
    model.to(device)
    print("Load Model Successful!")
    suffix = load_model_file.split('/')[-2]
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
            out = out[0, 0, ...].cpu().numpy()
            root_name = data["A_paths"][0].split("/")[-1][0:-4]
            target = 255 * (target / np.max(target))
            out = 255 * (out / np.max(out))

            # out[out >= 0.5] = 255
            # out[out < 0.5] = 0

            print('----------------------------------------------------------------------------------------------')
            print(os.path.join(save_root, "{}_lab.png".format(root_name)))
            print(os.path.join(save_root, "{}_pre.png".format(root_name)))
            print('----------------------------------------------------------------------------------------------')
            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)

    print("Finished!")
