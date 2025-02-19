from math import inf
import os
import time
import sys

import pdb
from termcolor import colored
import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer

def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self,use_amp=False):
        # 根据 use_amp 参数决定是否创建 AMP Scaler
        self.use_amp = use_amp
        if self.use_amp:
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None  # 如果不使用 AMP，则不需要 Scaler

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, retain_graph=False):
        # 根据 use_amp 决定是否使用 AMP 进行缩放和反向传播
        if self.use_amp:
            self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        else:
            loss.backward(create_graph=create_graph, retain_graph=retain_graph)

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                if self.use_amp:
                    self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                if self.use_amp:
                    self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
                
            if self.use_amp:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()
        else:
            norm = None
        
        return norm

    def state_dict(self):
        if self.use_amp:
            return self._scaler.state_dict()
        else:
            return {}

    def load_state_dict(self, state_dict):
        if self.use_amp:
            self._scaler.load_state_dict(state_dict)


def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)

    # Remove existing handlers if any
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False


    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

import csv
import os

class BlockLossRecorder:
    def __init__(self, file_path):
        """
        初始化 BlockLossRecorder
        :param file_path: 用于存储loss记录的文件路径
        """
        self.file_path = file_path
        self.loss_data = {}


    def record(self, blk_id, step, loss):
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

def plot_loss(block_loss_data, blk_id):
    """
    使用matplotlib绘制指定block的loss曲线
    :param block_loss_data: 某个block的step和loss数据
    :param blk_id: 要绘制的block ID
    """
    if not block_loss_data:
        print(f"没有找到 block {blk_id} 的 loss 数据")
        return

    steps, losses = zip(*block_loss_data)  # 解压出 step 和 loss
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label=f'Block {blk_id}', marker='o')

    # 设置标题和标签
    plt.title(f'Loss Curve for Block {blk_id}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# # 示例用法
# if __name__ == "__main__":
#     recorder = BlockLossRecorder("loss_records.csv")
    
#     # 记录一些数据
#     recorder.record("block1", 1, 0.5)
#     recorder.record("block1", 2, 0.4)
#     recorder.record("block2", 1, 0.6)
    
#     # 将数据保存到文件
#     recorder.save_to_file()
    
#     # 读取数据用于画图
#     loss_data = recorder.get_loss_data("block1")
#     print("block1的loss数据:", loss_data)
def get_tensor_size(tensor):
    """获取张量的内存大小，单位是字节"""
    return tensor.numel() * tensor.element_size()

def profile_memory(obj):
    """统计给定对象的内存占用，包括 nn.Module 和包含 Tensor 的其他对象"""
    total_size = 0
    device_memory = {}
    
    def process_tensor(tensor):
        """处理一个张量并更新总内存统计"""
        nonlocal total_size
        size = get_tensor_size(tensor)
        total_size += size
        device = tensor.device.type
        if device not in device_memory:
            device_memory[device] = 0
        device_memory[device] += size

    def process_object(obj):
        """递归处理对象及其属性"""
        if isinstance(obj, torch.nn.Module):
            # 处理 nn.Module 的参数和缓冲区
            for param in obj.parameters():
                process_tensor(param)
            for buffer in obj.buffers():
                process_tensor(buffer)
        elif isinstance(obj, torch.Tensor):
            # 如果对象本身就是张量
            process_tensor(obj)
        elif hasattr(obj, '__dict__'):
            # 处理其他对象，递归检查其属性
            for attr_name, attr_value in vars(obj).items():
                process_object(attr_value)
        elif isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
            # 处理容器类型
            for item in obj:
                process_object(item)
        elif isinstance(obj, dict):
            # 处理字典
            for key, value in obj.items():
                process_object(value)

    # 开始递归处理对象
    process_object(obj)

    # 计算结果
    total_mb = total_size / (1024 ** 2)
    device_memory_mb = {k: v / (1024 ** 2) for k, v in device_memory.items()}

    return {
        "total_memory_bytes": total_size,
        "total_memory_MB": total_mb,
        "device_memory_MB": device_memory_mb
    }