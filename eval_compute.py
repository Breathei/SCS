'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from thop import profile
import torch
import util.misc as utils
from main import get_args_parser
import argparse
from models.decoder import build

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()

if __name__ == '__main__':
    device = utils.validate_runtime_device(args.device)
    args.device = device
    model, _, = build(args)
    model.to(device)

    input = torch.randn(1, 3, 512, 512)
    samples = input.to(device)

    flops, params = profile(model, (samples, ))
    print("flops(G):", flops/1e9, "params(M):", params/1e6)
