import argparse
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import numpy as np
from EfficientQAT.utils import BlockLossRecorder

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
        type=str,
        help="path to model"
    )
    parser.add_argument(
        "--out_dir",
        default="/home/ubuntu/data/exp/proj2410/test/plots",
        type=str,
        help="path to output directory"
    )
    parser.add_argument(
        "--max_memory",
        default="24GB",
        type=str,
        help="maximum memory to use"
    )
    parser.add_argument(
        "--ppl_seqlen",
        default=2048,
        type=int,
        help="sequence length for perplexity evaluation"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="batch size for training"
    )
    parser.add_argument(
        "--plot_type",
        default="loss",
        type=str,
        help="type of plot to generate"
    )
    parser.add_argument(
        "--loss_path",
        default="/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast/loss.csv",
        type=str,
        help="path to loss file"
    )
    parser.add_argument(
        "--loss_contrast_path",
        default="/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-gradual-quant/loss.csv",
        type=str,
        help="path to contrast loss file"
    )
    args = parser.parse_args()
    return args

def plot_loss(loss_path,loss_contrast_path,args):
    recorder = BlockLossRecorder(loss_path)
    def drill_data(_recorder):
        data = []
        for i in _recorder.loss_data.keys():
            loss_data =_recorder.get_loss_data(i)
            _,loss_data = zip(*loss_data)
            data.append(loss_data)
        return data
    pivot_data = drill_data(recorder)
    recorder_contrast = BlockLossRecorder(loss_contrast_path)
    contrast_data = drill_data(recorder_contrast)
    out_dir = args.out_dir
    loss = np.array(pivot_data)
    contrast_loss = np.array(contrast_data)
    # Number of blocks to group in each plot
    blocks_per_plot = 2
    total_blocks = loss.shape[0]
    steps = loss.shape[1]

    # cols = total_blocks // blocks_per_plot // 2
    rows = total_blocks // blocks_per_plot // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3*rows))
    for i in range(total_blocks//blocks_per_plot):
        ax = axes[i//2,i%2]
        for blk in range(i*blocks_per_plot,(i+1)*blocks_per_plot):
            ax.plot(range(steps),loss[blk], label=f"block {blk}")
            ax.plot(range(steps),contrast_loss[blk], label=f"contrast block {blk}")
        ax.set_title(f'Loss for Blocks {i*blocks_per_plot } to {(i+1)*blocks_per_plot-1}')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_plot_interploate.pdf")

def main(args):
    if args.plot_type == "loss":
        plot_loss(loss_path=args.loss_path,
                  loss_contrast_path=args.loss_contrast_path,
                  args=args)
    pass

if __name__ == '__main__':
    args = args_parser()
    main(args)