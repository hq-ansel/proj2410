import torch.nn as nn
import torch.nn.functional as F
import torch

class Catcher(nn.Module):
    def __init__(self, module, dataset):
        super().__init__()
        self.module = module  # 包装的原始module
        self.dataset = dataset  # 用于更新输入数据的dataset
        self.index = 0  # 输入数据的索引
        self.attention_mask = None  # 用于存储attention mask
        self.position_ids = None  # 用于存储位置id
        self.stop_forward = False  # 控制前向传播的标志
        self.inps = {}  # 用于存储输入
        self.outs = {}  # 用于存储输出

    def forward(self, inp, **kwargs):
        # 存储输入和 kwargs 的内容
        combined_input = {"input": inp.squeeze(0).to('cpu')}
        combined_input.update({k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})
        
        # 将数据存入 dataset 和 inps
        self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
        self.inps[self.index] = combined_input  # 存储输入和 kwargs

        # 更新索引
        self.index += 1

        # 存储 attention_mask 和 position_ids
        if self.attention_mask is None:
            self.attention_mask = kwargs.get("attention_mask", None)
        if self.position_ids is None:
            self.position_ids = kwargs.get("position_ids", None)

        # 如果停止前向传播，抛出 ValueError（如果这是调试用途，可以移除）
        if self.stop_forward:
            raise ValueError

        # 前向传播，并存储输出
        output = self.module(inp, **kwargs)
        self.outs[self.index] = output  # 存储输出

        raise ValueError("Catcher should not be called during training")  # 调试用，可以移除

    def start_forward(self):
        # 允许前向传播
        self.stop_forward = False

    def stop_forward(self):
        # 阻止前向传播
        self.stop_forward = True

def train_single_layer(model, layer_idx, optimizer, input_data, target_model):
    # 解冻当前层，并冻结其它层
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.model.layers[layer_idx].named_parameters():
        param.requires_grad = True

    # 前向传播
    try:
        output = model(input_data)
    except ValueError:
        pass
    output = model.layers[layer_idx].outs[0] # outs[0] is out tensor
    with torch.no_grad():
        try:
            target_output = target_model(input_data)[0]
        except ValueError:
            pass
        target_output = target_model.layers[layer_idx].outs[0]

    # 获取当前层的自定义损失
    custom_loss = model.model.layers[layer_idx].get_custom_loss(target_output)

    # 标准损失函数，例如交叉熵损失
    standard_loss = nn.CrossEntropyLoss()(output, target_output)

    # 总损失 = 标准损失 + 自定义层损失
    total_loss = standard_loss + custom_loss

    # 反向传播和优化
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
def replace_layer_with_catcher(model, layer_idx):
    # 替换特定层为 Catcher 模块
    model.model.layers[layer_idx] = Catcher(model.model.layers[layer_idx])
    return model

def train_end2front(model, optimizer, input_data, target_model,args):
    for layer_idx in range(len(model.model.layers), 0, -1):
        model = replace_layer_with_catcher(model, layer_idx)
        target_model = replace_layer_with_catcher(target_model, layer_idx) 
        catcher_blk = model.model.layers[layer_idx]
        target_catcher_blk = target_model.model.layers[layer_idx]
        catcher_blk.start_forward()
        target_catcher_blk.start_forward()
        optimizer = torch.optim.Adam(catcher_blk.module.parameters(), lr=1e-5)

        for epoch in range(args.num_epochs):
            train_single_layer(model, layer_idx, optimizer, input_data, target_model)