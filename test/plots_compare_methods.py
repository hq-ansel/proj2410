import argparse
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from EfficientQAT.utils import BlockLossRecorder

def set_aaai_style():
    """设置符合AAAI会议要求的图片样式"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['font.style'] = 'normal'
    
    cmyk_colors = [
        (0, 0.7, 1, 0),  # 青色
        (1, 0, 0.7, 0),  # 品红色
        (1, 0.8, 0, 0),  # 黄色
        (0, 0, 0, 1),    # 黑色
    ]
     # 关键修改：强制 PDF 使用 CMYK
    mpl.rcParams['pdf.use14corefonts'] = True  # 避免字体嵌入问题
    mpl.rcParams['pdf.fonttype'] = 42          # 确保文本可编辑
    mpl.rcParams['ps.useafm'] = True           # 兼容 CMYK
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['savefig.transparent'] = False
    # 直接传递 CMYK 元组到颜色循环
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmyk_colors)

def args_parser():
    parser = argparse.ArgumentParser()
    # /home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128gradual/loss.csv
    parser.add_argument(
        "--w4_loss_path",
        # /home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128gradual_fator1/loss.csv
        default="/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128gradual/loss.csv",
        type=str,
        help="path to 4-bit loss file"
    )
    parser.add_argument(
        "--w2_loss_path",
        default="/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128/loss.csv",
        type=str,
        help="path to 2-bit loss file"
    )
    parser.add_argument(
        "--out_dir",
        default="/home/ubuntu/data/exp/proj2410/test/plots",
        type=str,
        help="path to output directory"
    )
    parser.add_argument(
        "--format",
        default="pdf",
        type=str,
        choices=["pdf", "png", "jpg"],
        help="output image format"
    )
    parser.add_argument(
        "--dpi",
        default=300,
        type=int,
        help="DPI for output image"
    )
    parser.add_argument(
        "--no_aaai_style",
        action="store_true",
        help="disable AAAI style formatting"
    )
    return parser.parse_args()

def load_loss_data(loss_path):
    """Load loss data from CSV file"""
    recorder = BlockLossRecorder(loss_path)
    recorder._load_from_file()
    data = []
    i="blk[3]"
    loss_data = recorder.get_loss_data(i)
    _, loss_data = zip(*loss_data)
    data.append(loss_data)
    return np.array(data)

def plot_bit_comparison(w4_loss_path, w2_loss_path, args):
    """Plot comparison between 4-bit and 2-bit training"""
    if not args.no_aaai_style:
        set_aaai_style()
    
    # Load data
    w4_loss = load_loss_data(w4_loss_path)
    w2_loss = load_loss_data(w2_loss_path)
    
    # Calculate average loss per step
    w4_avg = np.mean(w4_loss, axis=0)
    w2_avg = np.mean(w2_loss, axis=0)
    steps = np.arange(len(w4_avg))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    
    # 直接使用 CMYK 颜色
    ax1.plot(steps, w4_avg, label='PQ Average Loss', linewidth=2, 
             color="#FF4D00")  # 青色
    ax1.set_title('Progressive Quantization Training')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, w2_avg, label='One Shot Average Loss', linewidth=2, 
             color="#00FF4D")  # 品红
    ax2.set_title('One Shot Quantization Training')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存为 CMYK PDF
    output_file = f"{args.out_dir}/methods_comparison.pdf"
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    
    print(f"PDF 文件已保存至 {output_file}")
    plt.close()

def main(args):
    plot_bit_comparison(
        w4_loss_path=args.w4_loss_path,
        w2_loss_path=args.w2_loss_path,
        args=args
    )
# python -m test.plots_compare_methods
if __name__ == '__main__':
    args = args_parser()
    main(args)