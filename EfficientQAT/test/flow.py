import torch
import time

# 设备设置
device = "cuda:1"
import torch

# 模拟 40GB 数据（假设每个张量 100MB，共 400 个样本）
data_list = []
for _ in range(400):
    input_sample = torch.randn(100, 1000, 1000)  # 约 100MB
    output_sample = torch.randn(100, 1000, 1000)
    data_list.append((input_sample, output_sample))


# 流量控制参数
MAX_GPU_MEMORY_RATIO = 0.8  # GPU 内存占用阈值（80%）
MIN_SUBMIT_INTERVAL = 0.01  # 最小提交间隔（秒）
INITIAL_SUBMIT_INTERVAL = 0.1  # 初始提交间隔（秒）

# 动态调整提交间隔
submit_interval = INITIAL_SUBMIT_INTERVAL


for idx, sample in enumerate(data_list):
    # 检查 GPU 内存占用
    current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
    if current_memory > MAX_GPU_MEMORY_RATIO:
        submit_interval *= 2  # 内存占用过高，提交速度减半
        print(f"GPU 内存占用过高 ({current_memory:.2%})，提交间隔调整为 {submit_interval:.3f} 秒")
    else:
        submit_interval = max(submit_interval / 2, MIN_SUBMIT_INTERVAL)  # 恢复提交速度

    # 非阻塞数据传输
    input_sample, output_sample = sample
    input_sample = input_sample.to(device, non_blocking=True).unsqueeze(0)
    output_sample = output_sample.to(device, non_blocking=True).unsqueeze(0)

    # 异步计算
    with torch.no_grad():
        time.sleep(0.01)  # 模拟计算耗时

    # 异步将结果移回 CPU
    input_tensor = input_tensor.cpu()
    output_tensor = output_tensor.cpu()
    data_list[idx] = (input_tensor, output_tensor)

    # 控制提交速度
    time.sleep(submit_interval)