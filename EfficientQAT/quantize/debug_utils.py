import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn



logger = logging.getLogger(__name__)

def check_loss_anomaly(loss: torch.Tensor, threshold: float = 1000.0) -> bool:
    """
    检查loss是否异常
    
    Args:
        loss: 当前损失值
        threshold: 异常阈值
        
    Returns:
        bool: 如果loss异常返回True，否则返回False
    """
    if torch.isnan(loss) or torch.isinf(loss):
        return True
    
    if abs(loss.item()) > threshold:
        return True
    
    return False

def save_debug_info(model: nn.Module, 
                   inputs: Any,
                   outputs: Any,
                   loss: torch.Tensor,
                   gradients: Optional[List[torch.Tensor]] = None,
                   save_dir: str = "./debug_info",
                   prefix: str = "debug") -> None:
    """
    保存调试信息到磁盘
    
    Args:
        model: 当前模型
        inputs: 输入数据
        outputs: 输出数据
        loss: 损失值
        gradients: 梯度信息
        save_dir: 保存目录
        prefix: 文件前缀
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型状态
    model_path = os.path.join(save_dir, f"{prefix}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # 保存输入输出
    input_path = os.path.join(save_dir, f"{prefix}_inputs.pth")
    output_path = os.path.join(save_dir, f"{prefix}_outputs.pth")
    
    if isinstance(inputs, torch.Tensor):
        torch.save(inputs, input_path)
    elif isinstance(inputs, (list, tuple)):
        torch.save(inputs, input_path)
    else:
        # 尝试转换为tensor保存
        try:
            torch.save(inputs, input_path)
        except:
            with open(input_path.replace('.pth', '.pkl'), 'wb') as f:
                pickle.dump(inputs, f)
    
    if isinstance(outputs, torch.Tensor):
        torch.save(outputs, output_path)
    elif isinstance(outputs, (list, tuple)):
        torch.save(outputs, output_path)
    else:
        # 尝试转换为tensor保存
        try:
            torch.save(outputs, output_path)
        except:
            with open(output_path.replace('.pth', '.pkl'), 'wb') as f:
                pickle.dump(outputs, f)
    
    # 保存损失值
    loss_info = {
        "loss": loss.item(),
        "loss_tensor": loss
    }
    loss_path = os.path.join(save_dir, f"{prefix}_loss.pth")
    torch.save(loss_info, loss_path)
    
    # 保存梯度信息
    if gradients is not None:
        grad_path = os.path.join(save_dir, f"{prefix}_gradients.pth")
        torch.save(gradients, grad_path)
    
    # 保存模型结构信息
    model_info = {
        "model_class": model.__class__.__name__,
        "model_modules": [name for name, _ in model.named_modules()],
        "model_parameters": [name for name, _ in model.named_parameters()]
    }
    model_info_path = os.path.join(save_dir, f"{prefix}_model_info.json")
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Debug info saved to {save_dir}")

def collect_gradients(model: nn.Module) -> List[torch.Tensor]:
    """
    收集模型中所有参数的梯度
    
    Args:
        model: 模型
        
    Returns:
        List[torch.Tensor]: 梯度列表
    """
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.clone().detach())
        else:
            gradients.append(None)
    return gradients

def create_reproduction_script(save_dir: str, 
                              model_class: str,
                              input_shape: Tuple[int, ...],
                              device: str = "cuda") -> None:
    """
    创建可重现环境的脚本
    
    Args:
        save_dir: 保存目录
        model_class: 模型类名
        input_shape: 输入形状
        device: 设备
    """
    script_content = f'''#!/usr/bin/env python3
"""
Reproduction script for debugging loss anomaly
Generated automatically by debug_utils.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_debug_data():
    """加载调试数据"""
    save_dir = "{save_dir}"
    
    # 加载输入
    input_path = os.path.join(save_dir, "debug_inputs.pth")
    inputs = torch.load(input_path, map_location="{device}")
    
    # 加载模型
    model_path = os.path.join(save_dir, "debug_model.pth")
    # 注意：你需要根据具体情况初始化模型
    # model = {model_class}()
    # model.load_state_dict(torch.load(model_path, map_location="{device}"))
    
    return inputs

def main():
    print("Loading debug data...")
    inputs = load_debug_data()
    print(f"Inputs shape: {{inputs.shape if hasattr(inputs, 'shape') else 'N/A'}}")
    
    print("To debug the loss anomaly:")
    print("1. Initialize your model")
    print("2. Run forward pass with the loaded inputs")
    print("3. Check outputs and loss calculation")
    
if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(save_dir, "reproduce_debug.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Reproduction script saved to {script_path}")

def debug_model_state(model: nn.Module, 
                     inputs: Any,
                     outputs: Any,
                     loss: torch.Tensor,
                     save_dir: str = "./debug_info") -> None:
    """
    调试模型状态的综合函数
    
    Args:
        model: 模型
        inputs: 输入
        outputs: 输出
        loss: 损失值
        save_dir: 保存目录
    """
    logger.info("Debugging model state...")
    
    # 收集梯度
    gradients = collect_gradients(model)
    
    # 保存所有调试信息
    save_debug_info(
        model=model,
        inputs=inputs,
        outputs=outputs,
        loss=loss,
        gradients=gradients,
        save_dir=save_dir,
        prefix="debug"
    )
    
    # 创建重现脚本
    # create_reproduction_script(
    #     save_dir=save_dir,
    #     model_class=model.__class__.__name__,
    #     input_shape=inputs.shape if hasattr(inputs, 'shape') else (0,),
    #     device=next(model.parameters()).device if next(model.parameters(), None) is not None else 'cpu'
    # )

def check_tensor_anomalies(tensor: torch.Tensor, name: str = "") -> Dict[str, Any]:
    """
    检查张量中的异常值
    
    Args:
        tensor: 要检查的张量
        name: 张量名称
        
    Returns:
        Dict: 包含检查结果的字典
    """
    result = {
        "name": name,
        "shape": tensor.shape if hasattr(tensor, 'shape') else "N/A",
        "has_nan": torch.isnan(tensor).any().item() if isinstance(tensor, torch.Tensor) else False,
        "has_inf": torch.isinf(tensor).any().item() if isinstance(tensor, torch.Tensor) else False,
        "max_value": tensor.max().item() if isinstance(tensor, torch.Tensor) else None,
        "min_value": tensor.min().item() if isinstance(tensor, torch.Tensor) else None,
        "mean_value": tensor.mean().item() if isinstance(tensor, torch.Tensor) else None,
    }
    
    return result

def print_tensor_anomalies(tensor: torch.Tensor, name: str = "", threshold: float = 1000.0):
    """
    打印张量异常信息
    
    Args:
        tensor: 要检查的张量
        name: 张量名称
        threshold: 异常阈值
    """
    if not isinstance(tensor, torch.Tensor):
        logger.warning(f"{{name}} is not a tensor")
        return
        
    anomalies = check_tensor_anomalies(tensor, name)
    
    logger.info(f"Tensor {{name}} analysis:")
    for key, value in anomalies.items():
        logger.info(f"  {{key}}: {{value}}")
        
    # 检查是否超过阈值
    if abs(anomalies.get("max_value", 0)) > threshold or abs(anomalies.get("min_value", 0)) > threshold:
        logger.warning(f"Tensor {{name}} has values exceeding threshold {{threshold}}")