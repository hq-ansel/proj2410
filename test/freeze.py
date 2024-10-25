import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的三层网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.f1 = nn.Linear(10, 20)
        self.f2 = nn.Linear(20, 30)
        self.f3 = nn.Linear(30, 5)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return x

# 初始化模型和优化器
model = SimpleNet()

# 冻结 f1 和 f3 的参数
for param in model.f1.parameters():
    param.requires_grad = False

for param in model.f3.parameters():
    param.requires_grad = False

# 确保 f2 的参数是可训练的
for param in model.f2.parameters():
    param.requires_grad = True

# 使用随机输入和目标
input_data = torch.randn(32, 10)  # batch size 32, 输入特征大小 10
target = torch.randint(0, 5, (32,))  # 5 个类别的随机标签

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

# 前向传播（不使用 no_grad）
x1 = model.f1(input_data)  # f1 冻结
x2 = model.f2(x1)  # f2 可训练
output = model.f3(x2)  # f3 冻结

# 打印模型的参数梯度状态，调试用
for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")

# 打印损失前的检查
print("Output requires_grad:", output.requires_grad)

# 计算损失
loss = loss_function(output, target)

# 打印损失状态以确认是否有梯度路径
print(f"Loss requires grad: {loss.requires_grad}")
print(f"Loss grad_fn: {loss.grad_fn}")

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 验证 f2 的参数是否有梯度
print("f1.grad:", model.f1.weight.grad)  # 应该为 None
print("f2.grad:", model.f2.weight.grad)  # 应该有值
print("f3.grad:", model.f3.weight.grad)  # 应该为 None
