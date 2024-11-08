from typing import List, Union, Literal, Optional

import torch
import torch.nn as nn
import argparse
# from qtip.lib.codebook import get_id
# from qtip.lib.linear.quantized_linear import QuantizedLinear
from quip_sharp.lib.codebook import get_id
from quip_sharp.lib.linear.quantized_linear import QuantizedLinear as quip_sharp_QuantizedLinear
import bitblas
from EfficientQAT.quantize.int_linear_real import QuantLinear as EfficientQAT_QuantLinear

from bitblas.utils import auto_detect_nvidia_target  # noqa: E402
from bitblas.ops import MatmulConfig, Matmul  # noqa: E402

from gptqmodel.nn_modules.qlinear.qlinear_bitblas import BitBLASQuantLinear


def get_linear_layer(args):
    if args.quant_linear_implementation == 'quip_sharp':
        quip_param_dict = {
            2: {
                "codebook": "E8P12",
                "codebook_version": 1,
                "codesz": 8,
                "idx_dtype": "torch.int64",
                "lora_rank": 0,
                "packsz": 4,
                "rescale_WH": False,
                "resid_scale_override": -1
            },
            3: {
                "codebook": "E8P12RVQ3B",
                "codebook_version": 0,
                "codesz": 8,
                "idx_dtype": "torch.int64",
                "lora_rank": 0,
                "packsz": 2.6666666666666665,
                "rescale_WH": False,
                "resid_scale_override": -1
            },
            4: {
                "codebook": "E8P12RVQ4B",
                "codebook_version": 1,
                "codesz": 8,
                "idx_dtype": "torch.int64",
                "lora_rank": 0,
                "packsz": 2,
                "rescale_WH": False,
                "resid_scale_override": 3.6
            }
        }

        linear = quip_sharp_QuantizedLinear(
            args.in_channels,
            args.out_channels,
            # codebook_id=get_id(quip_param_dict[args.bits]['codebook']),
            codesz=quip_param_dict[args.bits]['codesz'],
            packsz=quip_param_dict[args.bits]['packsz'],
            pack_out=False,
            idx_dtype=quip_param_dict[args.bits]['idx_dtype'],
            codebook_version=quip_param_dict[args.bits]['codebook_version'],
            rank=quip_param_dict[args.bits]['lora_rank'],
            rescale_WH=quip_param_dict[args.bits]['rescale_WH'],
            resid_scale_override=quip_param_dict[args.bits]['resid_scale_override'],
            train_mode=False,
        )
        codebook_id=get_id(quip_param_dict[args.bits]['codebook'])
        linear.codebook_id.copy_(codebook_id)
    elif args.quant_linear_implementation == 'efficient_qat':
        bits = args.bits
        group_size = 128
        linear = EfficientQAT_QuantLinear(
            bits=bits,
            group_size=group_size,
            infeatures=args.in_channels,
            outfeatures=args.out_channels,
            bias=False,
        )
    elif args.quant_linear_implementation == 'gptq_bitblas':
        bits = args.bits
        group_size = 128
        linear = BitBLASQuantLinear(
            bits=bits,
            group_size=group_size,
            desc_act=False,
            sym=False,
            infeatures=args.in_channels,
            outfeatures=args.out_channels,
            bias=False,
            )
    return linear


def test_linear_time(linear, device='cuda'):
    in_features = getattr(linear, 'in_features', None)
    if in_features is None:
        in_features = linear.infeatures
    input_tensor = torch.randn(1, in_features).to(device=device, dtype=torch.float16)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(100):
        _ = linear(input_tensor)

    start_event.record()


    num_trials = 1000
    for _ in range(num_trials):
        output = linear(input_tensor)

    end_event.record()
    torch.cuda.synchronize()


    elapsed_time_ms = start_event.elapsed_time(end_event)
    average_time_us = elapsed_time_ms / num_trials * 1000 

    return average_time_us


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    linear = get_linear_layer(args).to(device)
    if args. quant_linear_implementation == 'gptq_bitblas':
        linear.post_init()
    average_time_us = test_linear_time(linear)
    print(f"{args.quant_linear_implementation} in_c {args.in_channels} out_c {args.out_channels} bits {args.bits} 平均每次计算时间: {average_time_us:.0f} 微秒")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_channels", type=int, default=4096)
    parser.add_argument("--out_channels", type=int, default=4096)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()
    from easydict import EasyDict as edict
    args = edict(vars(args))

    test_cases = [
        [4096, 4096],
        [4096, 11008],
        [11008, 4096],
        [5120, 5120],
        [5120, 13824],
        [13824, 5120],
        [8192, 8192],
        [8192, 28672],
        [28672, 8192]
    ]
    test_qlinear_implementations = ['quip_sharp', 'gptq_bitblas']
    for test_case in test_cases:
        for bits in [2,  4]:
            for qlinear_implementation in test_qlinear_implementations:
                args.in_channels = test_case[0]
                args.out_channels = test_case[1]
                args.bits = bits
                args.quant_linear_implementation = qlinear_implementation
                main(args)
