import torch
import torch.nn as nn
import torch.nn.functional as F

def MSE(output, target):
    """
    Mean Squared Error (MSE) loss.
    """
    return F.mse_loss(output, target)

# TODO: 手动实现cross block的损失函数函数

def FKLD(output, target):
    """
    Forward Kullback-Leibler divergence (FKLD).
    Assumes softmax is applied on both output and target.
    FKLD = KL(target || output)
    """
    output_prob = F.log_softmax(output, dim=-1)
    target_prob = F.softmax(target, dim=-1)
    return F.kl_div(output_prob, target_prob, reduction='batchmean')

def RKLD(output, target):
    """
    Reverse Kullback-Leibler divergence (RKLD).
    RKLD = KL(output || target)
    """
    output_prob = F.softmax(output, dim=-1)
    target_prob = F.softmax(target, dim=-1)
    return F.kl_div(target_prob.log(), output_prob, reduction='batchmean')

def FKLD_RKLD(output, target):
    """
    Combination of Forward and Reverse Kullback-Leibler divergences.
    """
    fkld = FKLD(output, target)
    rkld = RKLD(output, target)
    return fkld + rkld

def MSE_FKLD(output, target):
    """
    Combination of MSE and FKLD losses.
    """
    mse = MSE(output, target)
    fkld = FKLD(output, target)
    return mse + fkld

def MSE_RKLD(output, target):
    """
    Combination of MSE and RKLD losses.
    """
    mse = MSE(output, target)
    rkld = RKLD(output, target)
    return mse + rkld

def MSE_FKLD_RKLD(output, target):
    """
    Combination of MSE, FKLD, and RKLD losses.
    """
    mse = MSE(output, target)
    fkld_rkld = FKLD_RKLD(output, target)
    return mse + fkld_rkld

def MSE_MAX_ABS_ERROR(output, target):
    """
    Combination of MSE and max absolute error.
    """
    mse = MSE(output, target)
    max_abs_error = torch.max(torch.abs(output - target))
    return mse + max_abs_error*0.01

def low_frequency_loss(teacher_output, student_output, freq_cutoff=5):
    # 计算傅里叶变换
    teacher_freq = torch.fft.fft(teacher_output)
    student_freq = torch.fft.fft(student_output)
    
    # 截取低频成分
    teacher_low_freq = teacher_freq[:, :freq_cutoff]
    student_low_freq = student_freq[:, :freq_cutoff]
    
    # 计算低频损失
    # loss = torch.nn.functional.mse_loss(teacher_low_freq, student_low_freq)
    loss = torch.nn.functional.mse_loss(teacher_low_freq.real, student_low_freq.real)

    return loss

class AffineMSE(nn.Module):
    def __init__(self, label_dim):
        super(AffineMSE, self).__init__()
        # 初始化仿射变换矩阵 A，大小为 (label_dim, label_dim)
        self.A = nn.Parameter(torch.eye(label_dim))  # 初始化为单位矩阵

    def reinitialize_A(self):
        # 重初始化 A
        self.A.data = torch.eye(self.A.shape[0])
    def forward(self, out, label):
        # 使用 A 进行仿射变换
        transformed_label = label @ self.A
        # 计算损失
        loss = torch.mean((out - transformed_label) ** 2)+ F.mse_loss(out, transformed_label)*0.1
        return loss




def cross_block_loss(output, target):
    mse_part = F.mse_loss(output, target)
    # print(f"mse_part: {mse_part}")
    fkld_part = FKLD(output, target)
    # print(f"fkld_part: {fkld_part}")
    coef = 0.001
    return mse_part + fkld_part*coef


def get_loss_func(loss_type: str):
    """
    Get the loss function based on the loss type.
    """
    if loss_type == 'MSE':
        return MSE
    elif loss_type == 'AFFINE_MSE':
        return AffineMSE
    elif loss_type == 'FKLD':
        return FKLD
    elif loss_type == 'RKLD':
        return RKLD
    elif loss_type == 'FKLD_RKLD':
        return FKLD_RKLD
    elif loss_type == 'MSE_FKLD':
        return MSE_FKLD
    elif loss_type == 'MSE_RKLD':
        return MSE_RKLD
    elif loss_type == 'MSE_FKLD_RKLD':
        return MSE_FKLD_RKLD
    elif loss_type == 'MSE_MAX_ABS_ERROR':
        return MSE_MAX_ABS_ERROR
    elif loss_type == 'LOW_FOURIER_LOSS':
        return low_frequency_loss
    elif loss_type == 'CROSS_BLOCK_LOSS':
        return cross_block_loss
    else:
        raise ValueError(f'Invalid loss type: {loss_type}')