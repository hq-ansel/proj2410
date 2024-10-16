from transformers import Trainer, TrainerCallback
import torch
import torch.nn.functional as F
class CustomTrainCallback(TrainerCallback):
    def __init__(self, model, num_blocks=32):
        self.model = model
        self.num_blocks = num_blocks
        self.current_block = 0
        self.block_output = None  # 存储钩子提取的输出
        self.hook_handles = []
        
        # 冻结所有参数
        self.freeze_all_params()

    def freeze_all_params(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unlock_block(self, block_idx):
        for name, param in self.model.named_parameters():
            if f'block_{block_idx}' in name:  # 假设块命名为 block_x
                param.requires_grad = True

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.unlock_block(self.current_block)
        self.register_hooks_for_block(self.current_block)

    def register_hooks_for_block(self, block_idx):
        def hook_fn(module, input, output):
            self.block_output = output  # 保存当前块的输出
        self.remove_hooks()
        block_layer = getattr(self.model, f'block_{block_idx}')
        handle = block_layer.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def custom_loss_function(self, block_output):
        return torch.mean(block_output)  # 简单的损失示例

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % state.max_steps_per_epoch == 0:
            self.current_block = (self.current_block + 1) % self.num_blocks

class CustomTrainer(Trainer):
    def __init__(self, *args, callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback  # 用于传递回调
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 调用模型的forward方法获取输出
        outputs = model(**inputs)
        logits = outputs.logits  # 假设模型输出logits
        
        # 计算常规的交叉熵损失 (你可以替换成你需要的损失)
        labels = inputs.get("labels")
        main_loss = F.cross_entropy(logits, labels)
        
        # 使用 callback 提取的中间层输出来计算自定义损失
        if self.callback is not None and self.callback.block_output is not None:
            custom_loss = self.callback.custom_loss_function(self.callback.block_output)
            total_loss = main_loss + custom_loss  # 将自定义损失与主损失结合
        else:
            total_loss = main_loss

        return (total_loss, outputs) if return_outputs else total_loss


# 假设 model 已经实例化
callback = CustomTrainCallback(model)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callback=callback  # 传递 callback
)

trainer.train()
