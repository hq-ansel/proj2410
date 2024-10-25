import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ModelHandler:
    def __init__(self, model: nn.Module):
        self.model = model
    
    def get_module(self, module_name: str):
        module = dict(self.model.named_modules()).get(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found in the model.")
        return module

    def get_module_parameters(self, module_name: str):
        module = dict(self.model.named_modules()).get(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found in the model.")
        
        # 获取模块中所有参数
        return [param.clone() for param in module.parameters()]
    
    def set_module_parameters(self, module_name: str, new_parameters):
        module = dict(self.model.named_modules()).get(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found in the model.")
        
        # 设置模块中所有参数
        with torch.no_grad():
            for param, new_param in zip(module.parameters(), new_parameters):
                param.copy_(new_param)

class DirectionGenerator:
    @staticmethod
    def generate_random_directions(parameters):
        directions = []
        for param in parameters:
            direction = torch.randn_like(param)
            direction = direction / direction.norm() * param.norm()
            directions.append(direction)
        return directions

from EfficientQAT.quantize.utils import StopException,Catcher,set_op_by_name
from functools import wraps
import time
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

class LossSurfaceCalculator:
    def __init__(self, model_handler: ModelHandler,align_model: nn.Module, criterion: nn.Module):
        self.model_handler = model_handler
        self.align_model = align_model
        self.criterion = criterion
    
    def _calculate_loss(self, model,x,catch_model):
        device = next(model.parameters()).device
        with torch.no_grad():
            try:
                model(x.to(device))
            except StopException:
                pass
            except Exception as e:
                raise e
            out_tensor = catch_model.outs[0]
            try:
                self.align_model(x.to(device))
            except StopException:
                pass
            except Exception as e:
                raise e
            target_tensor = self.align_catch_model.outs[0]
            loss = self.criterion(out_tensor, target_tensor)
            return loss.item()
    @timer
    def calculate_loss_surface(self, module_name: str, dataloader: DataLoader, beta_range: np.ndarray):
        # 获取模块参数和方向向量
        theta_star = self.model_handler.get_module_parameters(module_name)
        d1 = DirectionGenerator.generate_random_directions(theta_star)
        d2 = DirectionGenerator.generate_random_directions(theta_star)
        
        catch_model = Catcher(self.model_handler.get_module(module_name),True)
        self.align_catch_model = Catcher(dict(self.align_model.named_modules()).get(module_name),True)
        set_op_by_name(self.model_handler.model,
                                        module_name,
                                        catch_model)
        set_op_by_name(self.align_model,
                                        module_name,
                                        self.align_catch_model)
        loss_surface = np.zeros((len(beta_range), len(beta_range)))
        
        for i, beta1 in enumerate(beta_range):
            for j, beta2 in enumerate(beta_range):
                # 根据扰动生成新的参数
                new_parameters = [
                    theta + beta1 * dir1 + beta2 * dir2
                    for theta, dir1, dir2 in zip(theta_star, d1, d2)
                ]
                self.model_handler.set_module_parameters(module_name, new_parameters)
                
                # 计算数据集上的平均损失
                total_loss = 0.0
                total_samples = 0
                with torch.no_grad():
                    for batch in dataloader:
                        loss= self._calculate_loss(self.model_handler.model,batch,catch_model)
                        total_loss += loss * batch.size(0)  # 累积样本数
                        total_samples += batch.size(0)
                
                # 计算平均损失
                # import pdb;pdb.set_trace()
                loss_surface[i, j] = total_loss / total_samples if total_samples > 0 else float('inf')
        set_op_by_name(self.model_handler.model,
                                        module_name,
                                        catch_model.module)
        set_op_by_name(self.align_model,
                                        module_name,
                                        self.align_catch_model.module)
        return loss_surface

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LossSurfaceVisualizer:
    @staticmethod
    def plot_3d_surface(beta_range: np.ndarray, loss_surface: np.ndarray,module_name: str):
        X, Y = np.meshgrid(beta_range, beta_range)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(X, Y, loss_surface, cmap='viridis')
        ax.set_xlabel('Beta 1')
        ax.set_ylabel('Beta 2')
        ax.set_zlabel('Loss')
        plt.title(f'Loss Landscape of {module_name}')
        plt.tight_layout()
        plt.savefig(f"/home/ubuntu/data/exp/proj2410/test/plots/loss_surface_{module_name}.pdf")

from torch.utils.data import Dataset, DataLoader
class SequenceDataset(Dataset):
    def __init__(self, data, seqlen):
        self.data = data
        self.seqlen = seqlen
        self.nsamples = data.numel() // seqlen
        
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, idx):
        start = idx * self.seqlen
        end = start + self.seqlen
        return self.data[:, start:end].squeeze(0)

from EfficientQAT.quantize.int_linear_real import load_quantized_model,QuantLinear
from EfficientQAT.datautils_block import get_loaders
from transformers import AutoTokenizer,AutoModelForCausalLM
def main():
    # 定义模型和损失函数
    quant_path = "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast"
    quant_model,tokenizer = load_quantized_model(quant_path,2,128)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # real to fake
    for name, module in quant_model.named_modules():
        if isinstance(module, QuantLinear):
            module.cuda()
            module.use_fake_quantization(del_quant=True,transpose=True)
            module.cpu()
    quant_model.to(device)

    testloader =get_loaders(name="wikitext2",
                               tokenizer=tokenizer,
                               train_size=128,
                               val_size=64,
                               seed=42,
                               seqlen=2048,
                               test_only=True)
    testenc = testloader.input_ids
    nsamples = testenc.numel() // 2048
    dataset = SequenceDataset(testenc[:, :128*2048], 2048)
    testloader = DataLoader(dataset, batch_size=64, shuffle=False)

    criterion = nn.MSELoss()
    model_handler = ModelHandler(quant_model)
    align_model_path = "/home/ubuntu/data/exp/proj2410/model/Qwen-2.5-0.5B"
    align_model = AutoModelForCausalLM.from_pretrained(align_model_path,torch_dtype=torch.float16)
    align_model.to(device)
    align_model.eval()
    loss_calculator = LossSurfaceCalculator(model_handler,align_model,criterion)
    visualizer = LossSurfaceVisualizer()

    # 设置beta范围
    beta_range = np.linspace(-1, 1, 50)

    test_moudule_name_list = [
        f"model.layers.{i}" for i in range(24)
    ]
    for module_name in test_moudule_name_list:
        # 计算损失曲面
        loss_surface = loss_calculator.calculate_loss_surface(module_name, testloader, beta_range)

        # 可视化
        visualizer.plot_3d_surface(beta_range, loss_surface, module_name)
main()