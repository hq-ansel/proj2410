import torch
import torch.nn.functional as F

def MSE(output, target):
    """
    Mean Squared Error (MSE) loss.
    """
    return F.mse_loss(output, target)

def FKLD(output, target):
    """
    Forward Kullback-Leibler divergence (FKLD).
    Assumes softmax is applied on both output and target.
    FKLD = KL(target || output)
    """
    output_prob = F.softmax(output, dim=-1)
    target_prob = F.softmax(target, dim=-1)
    return F.kl_div(output_prob.log(), target_prob, reduction='batchmean')

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

def get_loss_func(loss_type: str):
    """
    Get the loss function based on the loss type.
    """
    if loss_type == 'MSE':
        return MSE
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
    else:
        raise ValueError(f'Invalid loss type: {loss_type}')