import csv
import os

class BlockLossRecorder:
    def __init__(self, file_path:str):
        """
        初始化 BlockLossRecorder
        :param file_path: 用于存储loss记录的文件路径
        """
        self.file_path = file_path
        self.loss_data = {}

        # 如果文件存在，则从文件中加载已有的数据
        if os.path.exists(self.file_path):
            self._load_from_file()

    def record(self, blk_id:str, step:int, loss:float):
        """
        记录指定 block 和 step 的 loss 值
        :param blk_id: block 的 ID
        :param step: 当前 step
        :param loss: 当前 step 对应的 loss
        """
        if blk_id not in self.loss_data:
            self.loss_data[blk_id] = []
        self.loss_data[blk_id].append((step, loss))

    def save_to_file(self):
        """
        将记录的 loss 数据保存到文件
        """
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["blk_id", "step", "loss"])
            for blk_id, records in self.loss_data.items():
                for step, loss in records:
                    writer.writerow([blk_id, step, loss])

    def _load_from_file(self):
        """
        从文件中加载 loss 数据
        """
        with open(self.file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            for row in reader:
                blk_id, step, loss = row[0], int(row[1]), float(row[2])
                if blk_id not in self.loss_data:
                    self.loss_data[blk_id] = []
                self.loss_data[blk_id].append((step, loss))

    def get_loss_data(self, blk_id):
        """
        获取指定 block ID 对应的 loss 数据
        :param blk_id: block 的 ID
        :return: 对应的 step 和 loss 列表
        """
        return self.loss_data.get(blk_id, [])

import matplotlib.pyplot as plt
import numpy as np
def plot_loss(block_loss_data, blk_id, downsample_step=25, smoothing_window=25):
    """
    使用matplotlib绘制指定block的loss曲线，支持下采样和移动平均
    :param block_loss_data: 某个block的step和loss数据
    :param blk_id: 要绘制的block ID
    :param downsample_step: 下采样步长（每n个step显示一个点）
    :param smoothing_window: 移动平均的窗口大小
    """
    if not block_loss_data:
        print(f"没有找到 block {blk_id} 的 loss 数据")
        return

    steps, losses = zip(*block_loss_data)  # 解压出 step 和 loss

    # 进行下采样
    steps = steps[::downsample_step]
    losses = losses[::downsample_step]

    # 计算移动平均
    smooth_losses = np.convolve(losses, np.ones(smoothing_window)/smoothing_window, mode='valid')

    # 调整steps以匹配平滑后的loss长度
    smooth_steps = steps[:len(smooth_losses)]

    plt.figure(figsize=(10, 6))
    plt.plot(smooth_steps, smooth_losses, label=f'Block {blk_id} (Smoothed)', marker='o')
    # 使用散点图代替折线图
    # plt.scatter(smooth_steps, smooth_losses, label=f'Block {blk_id} (Smoothed)', marker='o')

    # 设置标题和标签
    plt.title(f'Loss Curve for Block {blk_id} (Smoothed)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# # 示例用法
# if __name__ == "__main__":
# loss_dir="/home/ubuntu/data/exp/proj2410/logs"
# file = os.path.join(loss_dir,"Llama2-7b-efficientqat-w2gs128.csv")
quant_model_path = "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128"
loss_path = f"{quant_model_path}/loss.csv"
recorder = BlockLossRecorder(loss_path)
# pdf_name = 

data=[]
for i in recorder.loss_data.keys():
    loss_data =recorder.get_loss_data(i)
    _,loss_data = zip(*loss_data)
    data.append(loss_data)
len(data),len(data[0])
min(data[0][:512]),min(data[0][512:1024]),min(data[0][1024:1536]),min(data[0][1536:2048]),min(data[0][2048:2560])


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

out_dir = "/home/ubuntu/data/exp/proj2410/test/"
loss = np.array(data)
# Number of blocks to group in each plot
blocks_per_plot = 4
total_blocks = loss.shape[0]
steps = loss.shape[1]

cols = total_blocks // blocks_per_plot // 2

# Calculate mean and threshold for outlier detection
mean_loss = np.mean(loss)
threshold = 4 * mean_loss

# Create a color map for different blocks
colors = plt.cm.tab10(np.linspace(0, 1, total_blocks))

# Plot the loss for each block group in 8 plots
fig, axs = plt.subplots(2, cols, figsize=(16, 8))  # Main plots

for i in range(total_blocks // blocks_per_plot):
    ax = axs[i // cols, i % cols]
    
    for blk in range(i * blocks_per_plot, (i + 1) * blocks_per_plot):
        # Identify outliers
        block_loss = loss[blk]
        outlier_indices = np.where(block_loss > threshold)[0]
        non_outlier_loss = np.clip(block_loss, None, threshold)
        
        # Assign color to the current block
        color = colors[blk]
        
        # Plot non-outliers in the main plot
        ax.plot(range(steps), non_outlier_loss, label=f'Block {blk+1}', color=color)
        
        # Add an inset to show outliers for this block only
        if len(outlier_indices) > 0:
            inset = inset_axes(ax, width="30%", height="30%", loc="upper right")
            inset.scatter(outlier_indices, block_loss[outlier_indices], color=color, s=10, label=f'Block {blk+1}')
            inset.set_title(f"Outliers (Block {blk+1})", fontsize=8)
            inset.set_xlim(0, steps)
            inset.set_ylim(threshold, np.max(block_loss) * 1.1)  # Show outlier range
            inset.tick_params(axis='both', which='major', labelsize=6)
            inset.legend(fontsize=6)

    ax.set_title(f'Loss for Blocks {i*blocks_per_plot + 1} to {(i+1)*blocks_per_plot}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(out_dir, f'{quant_model_path.split("/")[-1]}_with_inset.pdf'))