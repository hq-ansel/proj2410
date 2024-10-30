import torch
import argparse
# from qtip.lib.codebook import get_id
# from qtip.lib.linear.quantized_linear import QuantizedLinear
from quip_sharp.lib.codebook import get_id
from quip_sharp.lib.linear.quantized_linear import QuantizedLinear


def get_linear_layer(args):
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

    linear = QuantizedLinear(
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

    return linear


def test_linear_time(linear, device='cuda'):
    input_tensor = torch.randn(1, linear.in_features).to(device=device, dtype=torch.float16)

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
    average_time_us = test_linear_time(linear)
    print(f"in_c {args.in_channels} out_c {args.out_channels} bits {args.bits} 平均每次计算时间: {average_time_us:.0f} 微秒")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_channels", type=int, default=4096)
    parser.add_argument("--out_channels", type=int, default=4096)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

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

    for test_case in test_cases:
        for bits in [2, 3, 4]:
            args.in_channels = test_case[0]
            args.out_channels = test_case[1]
            args.bits = bits
            main(args)
