import os
from typing import Optional, Dict, Any
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class VisualizationRecorder:
    def __init__(self, visualization_type: str = "tensorboard", log_dir: str = "./logs", 
                 project_name: str = "efficientqat", experiment_name: str = "experiment"):
        """
        初始化可视化记录器
        
        Args:
            visualization_type: 可视化类型 ("tensorboard" 或 "wandb")
            log_dir: 日志目录路径 (用于 TensorBoard)
            project_name: 项目名称 (用于 wandb)
            experiment_name: 实验名称
        """
        self.visualization_type = visualization_type
        self.log_dir = log_dir
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        if visualization_type == "tensorboard" and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=log_dir)
        elif visualization_type == "wandb" and WANDB_AVAILABLE:
            wandb.init(project=project_name, name=experiment_name)
        else:
            self.writer = None
            print(f"Warning: {visualization_type} is not available. Please install the required packages.")
    
    def record_loss(self, blk_id: str, step: int, loss: float, 
                   loss_type: str = "train_loss"):
        """
        记录指定块的损失值
        
        Args:
            blk_id: 块的 ID (例如 "blk[0, 1, 2]" 或 "blk0")
            step: 当前步骤
            loss: 损失值
            loss_type: 损失类型 ("train_loss" 或 "val_loss")
        """
        tag = f"{loss_type}/{blk_id}"
        
        if self.visualization_type == "tensorboard" and TENSORBOARD_AVAILABLE:
            self.writer.add_scalar(tag, loss, step)
        elif self.visualization_type == "wandb" and WANDB_AVAILABLE:
            wandb.log({tag: loss, "step": step})
    
    def record_scalar(self, name: str, value: float, step: int):
        """
        记录标量值
        
        Args:
            name: 标量名称
            value: 标量值
            step: 当前步骤
        """
        if self.visualization_type == "tensorboard" and TENSORBOARD_AVAILABLE:
            self.writer.add_scalar(name, value, step)
        elif self.visualization_type == "wandb" and WANDB_AVAILABLE:
            wandb.log({name: value, "step": step})
    
    def close(self):
        """关闭记录器"""
        if self.visualization_type == "tensorboard" and TENSORBOARD_AVAILABLE:
            self.writer.close()
        elif self.visualization_type == "wandb" and WANDB_AVAILABLE:
            wandb.finish()