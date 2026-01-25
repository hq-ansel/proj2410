import torch
import triton
import triton.language as tl


# ---------------------------
# Autotune configs
# ---------------------------
def get_autotune_config_fwd():
    # Forward: weight packed along K, scales/zeros indexed by (N, K_group)
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
    ]


def get_autotune_config_bwd():
    # Backward-input: weight packed along N, scales/zeros indexed by (K, N_group)
    # Prefer BLOCK_N aligned with group_size (e.g., 128/256 when group_size=128).
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ]


# ---------------------------
# Forward kernel (optimized)
#   C = A @ Dequant(QW)^T (+ bias)
#   A: [M, K] (fp16/bf16)
#   qweight: [N, K//pack_factor] (u32/i32), packed along K
#   scales/qzeros: [N, ceil(K/group_size)]  (per-N per-Kgroup)
# ---------------------------
@triton.autotune(
    configs=get_autotune_config_fwd(),
    key=["M", "N", "K"],
)
@triton.jit
def int_matmul_kernel_opt(
    a_ptr,
    qweight_ptr,
    scales_ptr,
    qzeros_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_qwn,
    stride_qwk,
    stride_sn,
    stride_sk,
    stride_zn,
    stride_zk,
    stride_cm,
    stride_cn,
    n_bits: tl.constexpr,
    group_size: tl.constexpr,
    pack_factor: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # ---------------------------
    # Program ID mapping (same as your original grouping)
    # ---------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---------------------------
    # Block pointers
    # ---------------------------
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    qw_block_ptr = tl.make_block_ptr(
        base=qweight_ptr,
        shape=(N, K // pack_factor),
        strides=(stride_qwn, stride_qwk),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K // pack_factor),
        order=(1, 0),
    )

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Pack-unpack constants
    shifts_pack = (tl.arange(0, pack_factor) * n_bits).to(tl.uint32)  # [pack_factor]
    mask_val = (1 << n_bits) - 1

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # NOTE: This fast path assumes BLOCK_SIZE_K <= group_size (true for typical group_size=128, BK=32/64)
    # If you ever set group_size < BLOCK_SIZE_K, this would need per-column group_idx.
    for k_tile in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1)).to(OUT_DTYPE)
        qw_packed = tl.load(qw_block_ptr, boundary_check=(0, 1)).to(tl.uint32)  # (BN, BK/pack)

        # Unpack WITHOUT broadcast_to: (BN, BK/pack, pack) -> (BN, BK)
        w_int = (qw_packed[:, :, None] >> shifts_pack[None, None, :]) & mask_val
        w_int = tl.reshape(w_int, (BLOCK_SIZE_N, BLOCK_SIZE_K)).to(tl.int32)

        # Scales/Zeros for this K-group (constant across this BK tile)
        k_idx_base = k_tile * BLOCK_SIZE_K
        group_idx = k_idx_base // group_size

        s_ptr = scales_ptr + offs_bn * stride_sn + group_idx * stride_sk
        z_ptr = qzeros_ptr + offs_bn * stride_zn + group_idx * stride_zk

        scales = tl.load(s_ptr, mask=offs_bn < N, other=1.0).to(OUT_DTYPE)
        zeros = tl.load(z_ptr, mask=offs_bn < N, other=0).to(tl.int32)

        # Dequant in OUT_DTYPE (avoid large fp32 intermediates)
        w = (w_int - zeros[:, None]).to(OUT_DTYPE) * scales[:, None]  # (BN, BK)

        # Dot: (BM,BK) x (BK,BN) -> (BM,BN), accumulate fp32
        acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.float32, input_precision="ieee")

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        qw_block_ptr = tl.advance(qw_block_ptr, (0, BLOCK_SIZE_K // pack_factor))

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0).to(tl.float32)
        acc += b[None, :]

    # Store
    c = acc.to(OUT_DTYPE)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# ---------------------------
# Backward-input kernel (optimized)
#   dInput = dOut @ Dequant(QW)    (assuming dOut is [M, K], QW is [K, N] packed along N)
#
# EXPECTED LAYOUT (same as your original backward kernel):
#   grad_ptr: [M, K] (fp16/bf16)
#   qweight_ptr: [K, N//pack_factor] packed along N
#   scales/qzeros: [K, ceil(N/group_size)]  (per-K per-Ngroup)
# ---------------------------
@triton.autotune(
    configs=get_autotune_config_bwd(),
    key=["M", "N", "K"],
)
@triton.jit
def int_matmul_backward_input_kernel_opt(
    grad_ptr,
    qweight_ptr,
    scales_ptr,
    qzeros_ptr,
    d_input_ptr,
    M,
    N,
    K,
    stride_gm,
    stride_gk,
    stride_qwk,
    stride_qwn,
    stride_sn,
    stride_sk,
    stride_zn,
    stride_zk,
    stride_dim,
    stride_din,
    n_bits: tl.constexpr,
    group_size: tl.constexpr,
    pack_factor: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    g_block_ptr = tl.make_block_ptr(
        base=grad_ptr,
        shape=(M, K),
        strides=(stride_gm, stride_gk),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    qw_block_ptr = tl.make_block_ptr(
        base=qweight_ptr,
        shape=(K, N // pack_factor),
        strides=(stride_qwk, stride_qwn),
        offsets=(0, pid_n * (BLOCK_SIZE_N // pack_factor)),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N // pack_factor),
        order=(1, 0),
    )

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    shifts_pack = (tl.arange(0, pack_factor) * n_bits).to(tl.uint32)
    mask_val = (1 << n_bits) - 1

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # N groups for this tile (vector, to support BLOCK_N possibly spanning multiple groups)
    # group_idx_n: [BN]
    group_idx_n = (offs_bn // group_size).to(tl.int32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        g = tl.load(g_block_ptr, boundary_check=(0, 1)).to(OUT_DTYPE)
        qw_packed = tl.load(qw_block_ptr, boundary_check=(0, 1)).to(tl.uint32)  # (BK, BN/pack)

        # Unpack along N: (BK, BN/pack, pack) -> (BK, BN)
        w_int = (qw_packed[:, :, None] >> shifts_pack[None, None, :]) & mask_val
        w_int = tl.reshape(w_int, (BLOCK_SIZE_K, BLOCK_SIZE_N)).to(tl.int32)

        # Scales/Zeros are indexed by (K, N_group)
        k_indices = k_tile * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        # ptrs: (BK, BN)
        s_ptrs = scales_ptr + k_indices[:, None] * stride_sn + group_idx_n[None, :] * stride_sk
        z_ptrs = qzeros_ptr + k_indices[:, None] * stride_zn + group_idx_n[None, :] * stride_zk

        valid = (k_indices[:, None] < K) & (offs_bn[None, :] < N)
        scales = tl.load(s_ptrs, mask=valid, other=1.0).to(OUT_DTYPE)
        zeros = tl.load(z_ptrs, mask=valid, other=0).to(tl.int32)

        # Dequant in OUT_DTYPE
        w = (w_int - zeros).to(OUT_DTYPE) * scales  # (BK, BN)

        # Dot: (BM,BK) x (BK,BN) -> (BM,BN)
        acc = tl.dot(g, w, acc, out_dtype=tl.float32, input_precision="ieee")

        g_block_ptr = tl.advance(g_block_ptr, (0, BLOCK_SIZE_K))
        qw_block_ptr = tl.advance(qw_block_ptr, (BLOCK_SIZE_K, 0))

    out = acc.to(OUT_DTYPE)
    di_block_ptr = tl.make_block_ptr(
        base=d_input_ptr,
        shape=(M, N),
        strides=(stride_dim, stride_din),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(di_block_ptr, out, boundary_check=(0, 1))


# ---------------------------
# Python wrappers
# ---------------------------
def int_matmul_backend(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    g_idx=None,
    n_bits: int = 4,
    group_size: int = 128,
    **kwargs,
):
    """
    Forward:
      input:  (..., K) fp16/bf16
      qweight: (N, K//pack_factor) packed along K (uint32/int32)
      scales/qzeros: (N, ceil(K/group_size))

    Returns:
      (..., N)
    """
    x_shape = input.shape
    K = x_shape[-1]
    M = input.numel() // K
    x_2d = input.reshape(M, K)

    N = qweight.shape[0]
    pack_factor = 32 // n_bits

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    if input.dtype == torch.float16:
        OUT_DTYPE = tl.float16
    elif input.dtype == torch.bfloat16:
        OUT_DTYPE = tl.bfloat16
    else:
        OUT_DTYPE = tl.float32
    HAS_BIAS = bias is not None
    bias_ptr = bias if bias is not None else x_2d  # dummy pointer if no bias

    int_matmul_kernel_opt[grid](
        x_2d,
        qweight,
        scales,
        qzeros,
        bias_ptr,
        output,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        output.stride(0),
        output.stride(1),
        n_bits=n_bits,
        group_size=group_size,
        pack_factor=pack_factor,
        OUT_DTYPE=OUT_DTYPE,
        HAS_BIAS=HAS_BIAS,
    )

    return output.reshape(x_shape[:-1] + (N,))


def int_matmul_backward(
    grad_output: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx=None,
    n_bits: int = 4,
    group_size: int = 128,
    **kwargs,
):
    """
    Backward-input:
      grad_output: (..., K) fp16/bf16   (K = out_features)
      qweight: (K, N//pack_factor) packed along N
      scales/qzeros: (K, ceil(N/group_size))

    Returns:
      d_input: (..., N)

    IMPORTANT:
      这里的 qweight/scales/qzeros 的 layout 必须与上面说明一致（与你原始 backward kernel 一致）。
      如果你 forward 用的是 (N, K//pack) 的 qweight，那么 backward 需要提前准备对应的“转置打包”版本。
    """
    grad_shape = grad_output.shape
    K = grad_shape[-1]
    M = grad_output.numel() // K
    grad_2d = grad_output.reshape(M, K)

    pack_factor = 32 // n_bits
    N = qweight.shape[1] * pack_factor

    d_input = torch.empty((M, N), device=grad_output.device, dtype=grad_output.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    if grad_output.dtype == torch.float16:
        OUT_DTYPE = tl.float16
    elif grad_output.dtype == torch.bfloat16:
        OUT_DTYPE = tl.bfloat16
    else:
        OUT_DTYPE = tl.float32

    int_matmul_backward_input_kernel_opt[grid](
        grad_2d,
        qweight,
        scales,
        qzeros,
        d_input,
        M,
        N,
        K,
        grad_2d.stride(0),
        grad_2d.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        d_input.stride(0),
        d_input.stride(1),
        n_bits=n_bits,
        group_size=group_size,
        pack_factor=pack_factor,
        OUT_DTYPE=OUT_DTYPE,
    )

    return d_input.reshape(grad_shape[:-1] + (N,))


# Keep the same “backend.backward” convention you used
int_matmul_backend.backward = int_matmul_backward
