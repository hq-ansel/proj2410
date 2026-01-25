import torch
import triton
import triton.language as tl
import pdb

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,  # quantized matrix
    scales_ptr,  # scales, per group
    zeros_ptr,  # zeros, per group
    group_size,  # Should always be one of the supported group sizes
    result_ptr,  # Output matrix
    num_cols,  # input num cols in qweight
    num_rows,  # input num rows in qweight
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Setup the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (
        8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    )

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)

# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton(
                          input_tensor: torch.Tensor, # 输入张量
                          qweight: torch.Tensor, # 量化权重
                          scales: torch.Tensor, # 缩放因子
                          zeros: torch.Tensor, # 零点 
                          block_size_x: int = 32,
                          block_size_y: int = 32) -> torch.Tensor:
    # 从输入张量中提取维度信息：K 是 qweight 的行数，M 是 scales 的列数，group_size 是量化权重按组分块的大小
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]
    
    # 验证输入张量的形状和大小是否符合预期
    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K
    assert input_tensor.dtype==scales.dtype

    # 创建一个空的结果张量，其行数与 qweight 相同，但列数是 qweight 的 8 倍，
    # 并且存储在与 qweight 相同的设备上，类型与 scales 相同
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device=qweight.device,
                         dtype=scales.dtype)

    # qweight 的行数 Y 和列数 X
    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    # 定义一个 lambda 函数 grid 用于计算 Triton 内核的网格尺寸
    grid = lambda META: (
        triton.cdiv(X, META['BLOCK_SIZE_X']),
        triton.cdiv(Y, META['BLOCK_SIZE_Y']),
    )
    
    # 定义的 Triton 内核 awq_dequantize_kernel 进行去量化操作。这里 grid 参数指定内核的执行网格大小
    awq_dequantize_kernel[grid](qweight,
                                scales,
                                zeros,
                                group_size,
                                result,
                                X,
                                Y,
                                BLOCK_SIZE_X=block_size_x,
                                BLOCK_SIZE_Y=block_size_y)

    return torch.matmul(input_tensor, result)



@triton.jit
def awq_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    M,
    N,
    K,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # 获得程序入口指针 pid,piz
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N 向上取整除法
    # 得到分块后的id [0, BLOCK_SIZE_N-1] 属于id 0
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = c_ptr.type.element_ty

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # accumulator = tl.arange(0, BLOCK_SIZE_N)
    # accumulator = tl.broadcast_to(accumulator[None, :],
    # (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0
    # accumulator = accumulator.to(accumulator_dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.

    # [0,4]   +  [0 1 2 3].T
    # [[0,1,2,3].T [4,5,6,7].T] [[0,4] [1,5] [2,6], [3,7]]
    # 0, 4, 1, 5, 2, 6, 3, 7
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Create the necessary shifts to use to unpack.
    # 0, 16, 4, 20, 8, 24, 12, 28
    shifts = reverse_awq_order_tensor * 4
    # [[0, 16, 4, 20, 8, 24, 12, 28].T]
    # [K*N//8,8]
    # [K,N]
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    # 注意对于一个块 w 被压缩在了int32 解压的数量BLOCK_SIZE 对应 scale 为 BLOCK_SIZE//8
    # find block offsets for a and b
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bn = offsets_bn < N // 8

    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_zn = offsets_zn < N // 8

    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv
    # block_offset = BLOCK_SIZE_K * SPLIT_K
    # for k in range(0, (K + block_offset - 1) // (block_offset)):
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # Dequantize b.
        offsets_szk = (
            BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
        ) // group_size + tl.arange(0, 1)
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
# 定义一个名为 awq_gemm_triton 的函数，
# 输入张量 input、量化权重张量 qweight、比例因子张量 scales、零点张量 qzeros
# K 轴方向的分割迭代次数 split_k_iters。函数还接受三个可选参数，分别代表 M、N 和 K 方向上的块大小
def awq_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k_iters,
    )

    result = torch.zeros((M, N), dtype=scales.dtype, device=input.device)

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    awq_gemm_kernel[grid](
        input,
        qweight,
        result,
        qzeros,
        scales,
        M,
        N,
        K,
        group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        SPLIT_K=split_k_iters,
    )
    return result

# 4 bit 4/32=1/8
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['K'], # input-channel的维数
        x_vals=[256 * i for i in range(2, 21)], # input-channel的维数
        line_arg='provider',  # 线性化的维度
        line_vals=[
            "awq_dequantize_triton",
            "awq_gemm_triton",
        ], # label for the lines
        line_names=[
            "awq_dequantize_triton",
            "awq_gemm_triton",
        ],  # name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel='GB/s',  # y-axis label
        plot_name='kernel-performance',  # 图像名称
        args={'M': 4096,
              'group_size': 128, 
              "block_size_m": 128,
              "block_size_n": 128,
              "block_size_k": 64,
               'split_k_iters': 1,},  # 其他参数
    )
)
def benchmark_kernel(M, K, split_k_iters,
                     block_size_m, block_size_n, block_size_k,
                     group_size, provider):
    # M 为tokens数量 K为输入维度大小 G为组大小
    # qweights - [K     , M // 8], int32
    # scales   - [K // G, M     ], float16
    # zeros    - [K // G, M // 8], int32
    x = torch.randint(low=0, high=16, size=(M, K), device='cuda').to(torch.float16)
    qweight = torch.randint(low=0, high=16, size=(K, K // 8), device='cuda').to(torch.int32)
    scales = torch.rand(size=(K // group_size, K), device='cuda').to(torch.float16)
    qzeros = torch.randint(low=0, high=16, size=(K // group_size, K // 8), device='cuda').to(torch.int32)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'awq_gemm_triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: awq_gemm_triton(x, qweight, scales, qzeros, split_k_iters,
                                                                             block_size_m, block_size_n, block_size_k,
                                                                             ),quantiles=quantiles)
    elif provider == 'awq_dequantize_triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: awq_dequantize_triton(x,qweight, scales, qzeros,
                                                                                   block_size_m, block_size_n,
                                                                                   ),quantiles=quantiles)
    gbps =lambda ms: 2 * x.numel() * 4 / (ms * 1e6)
    return gbps(ms), gbps(min_ms), gbps(max_ms)
    # return ms, min_ms, max_ms

if __name__ == '__main__':
    benchmark_kernel.run(show_plots=False, print_data=True, save_path='.')