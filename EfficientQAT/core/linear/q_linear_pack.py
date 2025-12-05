import numpy as np
import torch
import transformers

from .q_linear_base import BaseQuantLinear


class PackableQuantLinear(BaseQuantLinear):
    def post_init(self, **kwargs):
        """
        Initialize weight factorization parameters for different bit-width quantization.
        
        This method sets up the weight factorization tensors (wf) based on the specified bit-width (2, 3, 4, or 8 bits).
        For 3-bit quantization, it uses a special pattern to handle the non-power-of-two case.
        
        Args:
            **kwargs: Additional keyword arguments passed to parent class's post_init method
        
        Attributes:
            wf_unsqueeze_zero (torch.Tensor): Weight factorization tensor with unsqueeze at dimension 0
            wf_unsqueeze_neg_one (torch.Tensor): Weight factorization tensor with unsqueeze at last dimension
        """
        super().post_init(**kwargs)

        if self.bits in [2, 4, 8]:
            wf = torch.tensor(list(range(0, self.pack_dtype_bits, self.bits)), dtype=torch.int32).unsqueeze(0).to(
                device=self.g_idx.device
            )
        elif self.bits == 3:
            wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12).to(device=self.g_idx.device)

        self.wf_unsqueeze_zero = wf.unsqueeze(0).to(device=self.g_idx.device)
        self.wf_unsqueeze_neg_one = wf.unsqueeze(-1).to(device=self.g_idx.device)

    def dequantize_weight(self, num_itr: int = 1):
        """
        将量化后的权重反量化为浮点数值，支持不同位宽(2/3/4/8 bits)的量化格式
        
        Args:
            num_itr (int): 迭代次数，默认为1。当大于1时，会将权重分块处理
        
        Returns:
            torch.Tensor: 反量化后的权重张量
        
        Note:
            对于3-bit量化有特殊处理逻辑，其他位宽(2/4/8 bits)使用统一处理流程
            当num_itr>1时，会分块计算权重以提高效率
        """
        if self.bits in [2, 4, 8]:
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
                self.wf_unsqueeze_zero,
            ).to(self.dequant_dtype)
            zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

            weight = torch.bitwise_and(
                torch.bitwise_right_shift(
                    torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                    self.wf_unsqueeze_neg_one,
                ).to(self.dequant_dtype),
                self.maxq,
            )
        elif self.bits == 3:
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(
                -1, -1, -1, 12
            )
            zeros = zeros >> self.wf_unsqueeze_zero
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = zeros & 0x7
            zeros = torch.cat(
                [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                dim=2,
            ).reshape(self.scales.shape)

            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
                -1, -1, 12, -1
            )
            weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        if num_itr == 1:
            weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim : (i + 1) * num_dim]
                weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim : (i + 1) * num_dim].long()
                weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
            weights = torch.cat(weights, dim=1)

        return weights

    def pack(self, linear: torch.nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor = None):
        """
        Pack the quantized weights and zeros into compressed format for storage.
        
        Args:
            linear (torch.nn.Module): The linear or conv layer to be packed
            scales (torch.Tensor): The quantization scales tensor
            zeros (torch.Tensor): The quantization zeros tensor
            g_idx (torch.Tensor, optional): The group indices for grouped quantization
        
        Notes:
            - Handles both linear and conv2d layers by flattening conv weights
            - Supports different bit widths (2,3,4,8 bits) for quantization
            - Stores packed weights in qweight and packed zeros in qzeros attributes
            - Converts tensors to numpy for efficient bit packing operations
        """
        W = linear.weight.data.clone()
        if isinstance(linear, torch.nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.T

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.T.contiguous()
        zeros = zeros.T.contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=torch.float16)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=torch.float16)

        int_weight = torch.round((W + scale_zeros[self.g_idx].T) / scales[self.g_idx].T).to(torch.int32)
        int_weight = int_weight.T.contiguous()
        int_weight = int_weight.numpy().astype(self.pack_np_math_dtype)

        qweight = np.zeros(
            (int_weight.shape[0] // self.pack_dtype_bits * self.bits, int_weight.shape[1]),
            dtype=self.pack_np_math_dtype,
        )
        if self.bits in [2, 4, 8]:
            for row in range(qweight.shape[0]):
                for j in range(self.pack_factor):
                    qweight[row] |= int_weight[row * self.pack_factor + j] << (self.bits * j)
        elif self.bits == 3:
            i = 0
            row = 0
            while row < qweight.shape[0]:
                for j in range(i, i + 10):
                    qweight[row] |= int_weight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= int_weight[i] << 30
                row += 1
                qweight[row] |= (int_weight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= int_weight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= int_weight[i] << 31
                row += 1
                qweight[row] |= (int_weight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= int_weight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1

        self.qweight = torch.from_numpy(qweight.astype(self.pack_np_dtype))

        zeros = zeros.numpy().astype(self.pack_np_math_dtype)
        qzeros = np.zeros(
            (zeros.shape[0], zeros.shape[1] // self.pack_dtype_bits * self.bits), dtype=self.pack_np_math_dtype
        )
        if self.bits in [2, 4, 8]:
            for col in range(qzeros.shape[1]):
                for j in range(self.pack_factor):
                    qzeros[:, col] |= zeros[:, col * self.pack_factor + j] << (self.bits * j)
        elif self.bits == 3:
            i = 0
            col = 0
            while col < qzeros.shape[1]:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1

        self.qzeros = torch.from_numpy(qzeros.astype(self.pack_np_dtype))


__all__ = ["PackableQuantLinear"]
