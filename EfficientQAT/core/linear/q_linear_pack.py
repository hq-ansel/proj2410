import numpy as np
import torch
import transformers
import torch.distributed as dist

try:
    from torch.distributed.device_mesh import DeviceMesh
except Exception:
    DeviceMesh = None

from .q_linear_base import BaseQuantLinear
from .q_linear_autograd import QuantLinearFunction

class PackableQuantLinear(BaseQuantLinear):
    def _unsigned_to_signed(self, x: torch.Tensor) -> torch.Tensor:
        sign_bit = 1 << (self.bits - 1)
        full_range = 1 << self.bits
        return torch.where(x >= sign_bit, x - full_range, x)

    def post_init(self, **kwargs):
        """
        初始化不同位宽量化的权重分解参数
        
        该方法根据指定的位宽（2、3、4 或 8 位）设置权重分解张量（wf）。
        对于 3 位量化，使用特殊模式来处理非 2 的幂次情况。
        
        Args:
            **kwargs: 传递给父类 post_init 方法的其他关键字参数
        
        Attributes:
            wf_unsqueeze_zero (torch.Tensor): 在第 0 维进行 unsqueeze 的权重分解张量
            wf_unsqueeze_neg_one (torch.Tensor): 在最后一维进行 unsqueeze 的权重分解张量
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
            torch.Tensor: 反量化后的权重张量，形状 [out_features, in_features]

        Note:
            对于3-bit量化有特殊处理逻辑，其他位宽(2/4/8 bits)使用统一处理流程
            当num_itr>1时，会分块计算权重以提高效率
        """
        if self.bits in [2, 4, 8]:
            # self.qzeros: 打包后的零点，形状 [in_features // group_size, out_features // pack_dtype_bits * bits]
            # unsqueeze: [in_features // group_size, 1, out_features // pack_dtype_bits * bits]
            # expand: [in_features // group_size, pack_factor, out_features // pack_dtype_bits * bits]
            # 移位并掩码以提取打包后的零点
            # zeros after expand: [in_features // group_size, pack_factor, out_features // pack_dtype_bits * bits]
            # zeros after reshape: [in_features // group_size, out_features]
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
                self.wf_unsqueeze_zero,
            )
            zeros = torch.bitwise_and(zeros, self.maxq).to(self.dequant_dtype).reshape(self.scales.shape)

            # self.qweight: 打包后的权重，形状 [out_features // pack_dtype_bits * bits, in_features]
            # unsqueeze: [out_features // pack_dtype_bits * bits, 1, in_features]
            # expand: [out_features // pack_dtype_bits * bits, pack_factor, in_features]
            # 移位并掩码以提取打包后的权重
            # weight: [out_features // pack_dtype_bits * bits, pack_factor, in_features]
            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
            )
            weight = torch.bitwise_and(weight, self.maxq).to(self.dequant_dtype)
        elif self.bits == 3:
            # 3-bit 非 2 的幂次位宽的特殊处理
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(
                -1, -1, -1, 12
            )
            zeros = zeros >> self.wf_unsqueeze_zero
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = zeros & 0x7
            # zeros after reshape: [in_features // group_size, out_features]
            zeros = torch.cat(
                [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                dim=2,
            ).reshape(self.scales.shape)

            # 3-bit 权重的特殊处理
            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
                -1, -1, 12, -1
            )
            weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            # weight after reshape: [out_features // pack_dtype_bits * bits, pack_factor, in_features]
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)

        # weight: 重塑为 [out_features, in_features]
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        if self.sym:
            weight = self._unsigned_to_signed(weight)

        if num_itr == 1:
            # self.scales: [in_features // group_size, out_features]
            # self.g_idx: [in_features] - 每个输入特征的组索引
            # self.g_idx.long(): [in_features]
            # scales[g_idx]: [out_features, in_features] - 将 scale 广播到每个位置
            # zeros[g_idx]: [out_features, in_features] - 将 zero 广播到每个位置
            # weights: [out_features, in_features]
            if self.sym:
                weights = self.scales[self.g_idx.long()] * weight
            else:
                weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                # scale_i: scale 的切片, [in_features // group_size, num_dim]
                scale_i = self.scales[:, i * num_dim : (i + 1) * num_dim]
                # weight_i: weight 的切片, [out_features, num_dim]
                weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
                # zeros_i: zero 的切片, [out_features, num_dim]
                zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
                # g_idx_i: 组索引的切片, [num_dim]
                g_idx_i = self.g_idx[i * num_dim : (i + 1) * num_dim].long()
                # scale_i[g_idx_i]: [out_features, num_dim] - 按组索引
                # zeros_i[g_idx_i]: [out_features, num_dim] - 按组索引
                if self.sym:
                    weights.append(scale_i[g_idx_i] * weight_i)
                else:
                    weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
            # weights: 连接所有切片, [out_features, in_features]
            weights = torch.cat(weights, dim=1)

        return weights

    def pack(self, linear: torch.nn.Module, scales: torch.Tensor, zeros: torch.Tensor | None, g_idx: torch.Tensor = None):
        """
        将量化后的权重和零点打包为压缩格式以便存储

        Args:
            linear (torch.nn.Module): 要打包的线性层或卷积层
                - Linear: 权重形状 [out_features, in_features]
                - Conv2d: 权重形状 [out_channels, in_channels, kernel_h, kernel_w]
                - Conv1D: 权重形状 [in_features, out_features] (转置格式)
            scales (torch.Tensor): 量化 scale 张量，形状 [out_features, in_features // group_size]
        zeros (torch.Tensor | None): 量化 zero 张量，形状 [out_features, in_features // group_size]
            对称量化 (sym=True) 时可为 None，将使用全 0 zero_point。
            g_idx (torch.Tensor, optional): 分组量化的组索引，形状 [in_features]

        Notes:
            - 通过展平卷积权重来处理线性层和卷积层
            - 支持不同的量化位宽（2、3、4、8 位）
            - 将打包后的权重存储在 qweight 属性中，打包后的零点存储在 qzeros 属性中
            - 将张量转换为 numpy 以进行高效的位打包操作
        """
        # W: 权重张量，处理后形状为: [out_features, in_features]
        W = linear.weight.data.clone()
        if isinstance(linear, torch.nn.Conv2d):
            W = W.flatten(1)  # [out_channels, in_channels, kernel_h, kernel_w] -> [out_channels, in_channels * kernel_h * kernel_w]
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.T  # [in_features, out_features] -> [out_features, in_features]

        # g_idx: 组索引映射，形状 [in_features]
        # torch.tensor([i // self.group_size for i in range(in_features)]
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        # scales: 从 [out_features, in_features // group_size] 转置
        #  为 [in_features // group_size, out_features]
        # zeros: 从 [out_features, in_features // group_size] 转置
        #  为 [in_features // group_size, out_features]
        scales = scales.T.contiguous()
        if zeros is None:
            if not self.sym:
                raise ValueError("zeros must be provided for asymmetric quantization.")
            zeros = torch.zeros_like(scales)
        zeros = zeros.T.contiguous()

        # scale_zeros: 零点 * scale，形状 [in_features // group_size, out_features]
        scale_zeros = zeros * scales

        # self.scales: 存储的 scale，形状 [in_features // group_size, out_features]
        self.scales = scales.clone().to(dtype=torch.float16)

        # self.bias: 偏置张量（如果存在），形状 [out_features]
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=torch.float16)

        # int_weight: 量化后的整数权重，形状 [out_features, in_features]
        # 公式: round((W + scale_zeros[g_idx].T) / scales[g_idx].T)
        # scales[g_idx].T: [in_features, out_features] - 
        # 将 scale 广播到每个权重位置
        # scale_zeros[g_idx].T: [in_features, out_features] - 
        # 将 scale_zeros 广播到每个权重位置
        # int_weight = torch.round((W + scale_zeros[self.g_idx].T)
        #                           / scales[self.g_idx].T).to(torch.int32)
        # int_weight = torch.round(W/ scales[self.g_idx].T+torch.round(zeros[self.g_idx]).T).to(torch.int32)
        int_weight = torch.round(W / scales[self.g_idx].T).to(torch.int32)
        if self.sym:
            int_weight = int_weight.clamp(self.qmin, self.qmax)
            int_weight = torch.bitwise_and(int_weight, self.maxq)
        else:
            int_weight = int_weight + torch.round(zeros[self.g_idx]).T.to(torch.int32)
            # int_weight: 限制在有效量化范围 [0, maxq] 内，
            #  形状 [out_features, in_features]
            int_weight = int_weight.clamp(0, self.maxq)
        if getattr(self, "debug_int_weight", False):
            self.int_weight_debug = int_weight.detach().cpu()
        elif hasattr(self, "int_weight_debug"):
            self.int_weight_debug = None

        # int_weight: 为打包进行转置，形状 [out_features, in_features]
        #  -> [in_features, out_features]
        int_weight = int_weight.T.contiguous()

        # int_weight: 转换为 numpy 以进行位打包，
        # 形状 [in_features, out_features]
        int_weight = int_weight.numpy().astype(self.pack_np_math_dtype)

        # qweight: 打包后的权重（压缩格式），形状 [out_features // pack_dtype_bits * bits, in_features]
        # 4-bit + int32 示例: [out_features // 8, in_features]
        # 每个 int32 打包 8 个 4-bit 值
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

        # self.qweight: packed quantized weights as torch tensor, shape [out_features // pack_dtype_bits * bits, in_features]
        self.qweight = torch.from_numpy(qweight.astype(self.pack_np_dtype))

        # zeros: convert to numpy for bit packing, shape [in_features // group_size, out_features]
        zeros = zeros.numpy().astype(self.pack_np_math_dtype)

        # qzeros: packed zero points (compressed), shape [in_features // group_size, out_features // pack_dtype_bits * bits]
        # Example for 4-bit + int32: [in_features // group_size, out_features // 8]
        # Each int32 packs 8 4-bit zero points
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

        # self.qzeros: 打包后的量化零点（torch 张量），形状 [in_features // group_size, out_features // pack_dtype_bits * bits]
        self.qzeros = torch.from_numpy(qzeros.astype(self.pack_np_dtype))

    def set_tp_mesh(
        self,
        tp_mesh,
        tp_mode: str,
        tp_dim=0,
        gather_output: bool = False,
        input_is_parallel: bool = False,
    ):
        """
        tp_mode:
          - "col": shard out_features（列并行 / column-parallel）
          - "row": shard in_features（行并行 / row-parallel）

        gather_output:
          - col: True  -> all_gather 拼回全量输出
          - col: False -> 输出保持 sharded（下一层按 Megatron 继续）
        input_is_parallel:
          - row: True  -> 输入已经按 last-dim 做过 shard（典型：上一层 col 不 gather）
          - row: False -> 本层自己切输入（调试/最小改动场景）
        """
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed 未初始化，先 dist.init_process_group(...)")

        # --- extract tp_group / rank / world_size ---
        if DeviceMesh is not None and isinstance(tp_mesh, DeviceMesh):
            tp_group = tp_mesh.get_group(tp_dim)
        else:
            tp_group = tp_mesh

        self.tp_group = tp_group
        self.tp_world_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        self.tp_mode = tp_mode
        self.tp_gather_output = gather_output
        self.tp_input_is_parallel = input_is_parallel

        # 单卡无需 shard
        if self.tp_world_size == 1:
            return

        # 你现在的 Triton dequant 仅支持 2/4/8
        if self.bits == 3:
            raise NotImplementedError("当前 Triton dequant 不支持 3-bit；需要单独 kernel 或回退 torch dequant。")

        # --- shard buffers ---
        if tp_mode == "col":
            self._tp_shard_out_features()
        elif tp_mode == "row":
            self._tp_shard_in_features()
        else:
            raise ValueError(f"Unknown tp_mode={tp_mode}, expected 'col' or 'row'.")

    def _tp_shard_out_features(self):
        """Column-parallel: shard out_features (dim=1 of qweight/scales, dim=0 of bias)"""
        tp = self.tp_world_size
        r = self.tp_rank

        if self.out_features % tp != 0:
            raise ValueError(f"out_features={self.out_features} not divisible by tp={tp}")

        local_out = self.out_features // tp
        start = r * local_out
        end = start + local_out

        # 对 qzeros：packed in out_features dimension，需要 pack_factor 对齐
        pf = self.pack_factor
        if (start % pf) != 0 or (local_out % pf) != 0:
            raise ValueError(
                f"Column TP requires out shard aligned to pack_factor={pf}, "
                f"but got start={start}, local_out={local_out}"
            )

        # scales: [num_groups, out_features]
        self.scales = self.scales[:, start:end].contiguous()
        # qweight: [in_features_packed, out_features]
        self.qweight = self.qweight[:, start:end].contiguous()
        # qzeros: [num_groups, out_features_packed]
        self.qzeros = self.qzeros[:, (start // pf):(end // pf)].contiguous()
        # bias: [out_features]
        if getattr(self, "bias", None) is not None:
            self.bias = self.bias[start:end].contiguous()

        # 更新元信息
        self.out_features = local_out

    def _tp_shard_in_features(self):
        """Row-parallel: shard in_features (dim=0 of g_idx, packed dim=0 of qweight)"""
        tp = self.tp_world_size
        r = self.tp_rank

        if self.in_features % tp != 0:
            raise ValueError(f"in_features={self.in_features} not divisible by tp={tp}")

        local_in = self.in_features // tp
        start = r * local_in
        end = start + local_in

        pf = self.pack_factor
        if (start % pf) != 0 or (local_in % pf) != 0:
            raise ValueError(
                f"Row TP requires in shard aligned to pack_factor={pf}, "
                f"but got start={start}, local_in={local_in}"
            )

        # g_idx: [in_features] -> slice
        self.g_idx = self.g_idx[start:end].contiguous()

        # qweight: [in_features_packed, out_features] -> slice packed rows
        p0 = start // pf
        p1 = end // pf
        self.qweight = self.qweight[p0:p1, :].contiguous()

        # scales/qzeros：为了兼容"任意 g_idx（可能是 permuted）"，这里先不 shard（复制开销很小）
        # 若你确认 g_idx 是单调且 shard 边界按 group_size 对齐，可进一步 shard scales/qzeros 来省一点点显存。

        self.in_features = local_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TP-aware forward.
        假设 x shape [..., in_features]
        """
        # Row-parallel 且输入未分片：本层自己切 x
        if getattr(self, "tp_world_size", 1) > 1 and getattr(self, "tp_mode", None) == "row":
            if not getattr(self, "tp_input_is_parallel", False):
                local_in = self.in_features
                start = self.tp_rank * local_in
                end = start + local_in
                x = x[..., start:end].contiguous()

        out_local = QuantLinearFunction.apply(
            x,
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.pack_dtype_bits,
            self.maxq,
            self.sym,
        )
        if getattr(self, "bias", None) is not None:
            out_local = out_local + self.bias

        # no TP
        if getattr(self, "tp_world_size", 1) == 1:
            return out_local

        # Column-parallel: optionally all_gather to full output
        if self.tp_mode == "col":
            if self.tp_gather_output:
                parts = [torch.empty_like(out_local) for _ in range(self.tp_world_size)]
                dist.all_gather(parts, out_local, group=self.tp_group)
                return torch.cat(parts, dim=-1)
            return out_local

        # Row-parallel: all_reduce sum to full output
        if self.tp_mode == "row":
            dist.all_reduce(out_local, op=dist.ReduceOp.SUM, group=self.tp_group)
            return out_local

        return out_local

def apply_tp(model, tp_mesh):
    for name, m in model.named_modules():
        if isinstance(m, PackableQuantLinear):
            # 典型 Megatron 策略（示例）：
            # attention: q/k/v & mlp up/gate -> col
            # attention o_proj & mlp down -> row
            if name.endswith(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
                m.set_tp_mesh(tp_mesh, tp_mode="col", tp_dim="tp", gather_output=False)
            elif name.endswith(("o_proj", "down_proj")):
                m.set_tp_mesh(tp_mesh, tp_mode="row", tp_dim="tp", input_is_parallel=True)
            else:
                # 不确定就先 col 并 gather，保证功能正确（但通信多）
                m.set_tp_mesh(tp_mesh, tp_mode="col", tp_dim="tp", gather_output=True)


__all__ = ["PackableQuantLinear"]
