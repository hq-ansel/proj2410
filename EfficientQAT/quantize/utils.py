from collections import OrderedDict
from .int_linear_fake import QuantLinear
import torch
from torch import nn
from typing  import Optional

class Catcher(nn.Module):
    """
    Args:
        module: nn.Module 需要包装的module
        dataset: Dataset 用于更新输入数据的dataset
        index: int 输入数据的索引
        attention_mask: Optional[torch.Tensor] 用于存储attention mask
        position_ids: Optional[torch.LongTensor] 用于存储位置id
        stop_forward: bool 控制前向传播的标志
        inps: dict 用于存储输入以及参数
        outs: dict 用于存储输出，Tuple类型
    """
    def __init__(self, module,stop_forward_flag=False):
        super().__init__()
        self.module = module  # 包装的原始module
        self.index = 0  # 输入数据的索引
        self.attention_mask = None  # 用于存储attention mask
        self.position_ids = None  # 用于存储位置id
        self.stop_forward_flag = stop_forward_flag  # 控制前向传播的标志
        self.inps = {}  # 用于存储输入
        self.outs=None
    def forward(self, inp, **kwargs):
        
        
        # 前向传播，并存储输出
        output = self.module(inp, **kwargs)
        self.outs = output  # 存储输出
        if self.stop_forward_flag:
            raise ValueError(f"stop forward shape: {output[0].shape}")
        return output

    def set_forward_state(self, stop_forward: bool):
        self.stop_forward_flag = stop_forward


class MultiBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_list = nn.ModuleList([])
    
    def add_block(self, block):
        self.block_list.append(block)
        
    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None):
        for block in self.block_list:
            hidden_states = block(hidden_states, attention_mask=attention_mask,position_ids=position_ids)[0]
        return (hidden_states, )


def set_weight_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
            m.requires_grad = requires_grad
    return iter(params)

def weight_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
            params.append(m)
    return iter(params)

def set_quant_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find('scale') > -1 or n.find('zero_point') > -1:
            m.requires_grad = requires_grad
    return iter(params)  

def quant_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('scale') > -1 or n.find('zero_point') > -1:
            params.append(m)
    return iter(params)  


def trainable_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if m.requires_grad:
            params.append(m)
    return iter(params)  

def trainable_parameters_num(model):
    params = []
    total = 0
    for n, m in model.named_parameters():
        if m.requires_grad:
            total += m.numel()
    return total

def set_quant_state(model, weight_quant: bool = False):
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.set_quant_state(weight_quant)
            
@torch.no_grad()   
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight.data = module.weight_quantizer(module.weight.data)


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     


def get_named_linears(module, type):
    # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)
        
# def add_new_module(name, original_module, added_module):
#     levels = name.split('.')
#     if len(levels) > 1:
#         mod_ = original_module
#         for l_idx in range(len(levels)-1):
#             if levels[l_idx].isdigit():
#                 mod_ = mod_[int(levels[l_idx])]
#             else:
#                 mod_ = getattr(mod_, levels[l_idx])
#         setattr(mod_, levels[-1], added_module)
#     else:
#         setattr(original_module, name, added_module)   

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


import torch
import gc

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