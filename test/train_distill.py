import torch
import torch.nn.functional as F
from accelerate import Accelerator

# 初始化加速器
accelerator = Accelerator()

def distillation_loss(student_output, teacher_output, true_labels, temperature=1.0, alpha=0.5):
    """
    知识蒸馏损失函数
    :param student_output: 学生模型的输出
    :param teacher_output: 教师模型的输出
    :param true_labels: 真实标签
    :param temperature: 温度系数
    :param alpha: 蒸馏损失的权重
    :return: 蒸馏损失
    """
    # 蒸馏损失 (使用KL散度)
    distillation_loss = F.kl_div(
        F.log_softmax(student_output / temperature, dim=-1),
        F.softmax(teacher_output / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    
    # 标准交叉熵损失
    true_loss = F.cross_entropy(student_output, true_labels)
    
    # 总损失
    return alpha * distillation_loss + (1.0 - alpha) * true_loss

def get_grad_norm(model):
    """
    计算模型梯度的整体 L2 范数
    """
    norms = []
    for param in model.parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.grad.detach(), p=2)
            norms.append(param_norm)
    
    overall_norm = torch.norm(torch.stack(norms), p=2)
    return overall_norm.item()

def validate(model, valloader):
    """
    评估模型在验证集上的性能
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in valloader:
            inputs, labels = batch["input_ids"], batch["labels"]
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(valloader)
    print(f"Validation Loss: {avg_loss}")

def inner_loop_train(student_model, teacher_model, train_dataloader, valloader, optimizer, num_epochs):
    """
    内循环训练函数
    """
    for epoch in range(num_epochs):
        student_model.train()
        for batch in train_dataloader:
            inputs, labels = batch["input_ids"], batch["labels"]

            # 通过教师模型生成教师的输出
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
            
            # 通过学生模型生成学生的输出
            student_output = student_model(inputs)

            # 计算蒸馏损失
            loss = distillation_loss(student_output, teacher_output, labels)

            # 反向传播和优化
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        # 在每个 epoch 结束时计算梯度的整体 L2 范数
        grad_norm = get_grad_norm(student_model)
        print(f"Epoch {epoch+1}/{num_epochs}, Gradient Norm: {grad_norm}")

        # 验证模型
        validate(student_model, valloader)

# 主训练流程
if __name__ == "__main__":
    # 假设模型、数据加载器和优化器已初始化
    teacher_model = ...  # 定义或加载教师模型
    student_model = ...  # 定义或加载学生模型

    # 初始化数据加载器和优化器
    train_dataloader = ...  # 训练数据集的 DataLoader
    valloader = ...  # 验证数据集的 DataLoader

    # 外循环：动态生成优化器并执行内循环训练
    num_outer_loops = 5
    num_epochs_per_loop = 10
    for i in range(num_outer_loops):
        # 动态生成优化器
        optimizer = customOptimizer(student_model, config_for_outer_loop[i])
        
        # 将新的优化器移到设备
        optimizer = accelerator.prepare(optimizer)
        train_dataloader, valloader = accelerator.prepare(train_dataloader, valloader)
        
        # 执行内循环训练
        inner_loop_train(student_model, teacher_model, train_dataloader, valloader, optimizer, num_epochs_per_loop)
