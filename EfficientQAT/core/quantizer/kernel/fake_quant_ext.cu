#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <stdint.h>
#include <math.h>

namespace cg = cooperative_groups;

// --- Helper Macros for Tensor Validation ---
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, t) TORCH_CHECK(x.scalar_type() == t, #x " has wrong dtype")
#define CHECK_SHAPE(cond, msg) TORCH_CHECK(cond, msg)

// --- Math Utilities ---
// Bankers rounding (round to nearest even) to match standard quantization behavior
__device__ __forceinline__ float round_bankers(float x) { return nearbyintf(x); }

// Standard integer clamping
__device__ __forceinline__ int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// clamp(abs(x)) * sign(x) as in BaseQuantizer.cal_qparams (STE clamp)
__device__ __forceinline__ float clamp_abs_sign(float x, float lo, float hi) {
    float ax = fabsf(x);
    float clamped = fminf(fmaxf(ax, lo), hi);
    return copysignf(clamped, x);
}

// --- Type Conversion Traits ---
// Used to handle float, fp16, and bf16 generically in kernels
template<typename T> __device__ __forceinline__ float to_f32(T);
template<> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template<> __device__ __forceinline__ float to_f32<half>(half v) { return __half2float(v); }
template<> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ __forceinline__ T from_f32(float);
template<> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template<> __device__ __forceinline__ half from_f32<half>(float v) { return __float2half_rn(v); }
template<> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template <typename T>
struct TypeTag { using type = T; };

// --- Persistent Work Queue Helpers ---
__device__ __forceinline__ int persistent_next(int* counter) {
    // 1 CTA fetches 1 task (group)
    return atomicAdd(counter, 1);
}

template<int CHUNK>
__device__ __forceinline__ int persistent_next_chunk(int* counter) {
    // Optional: fetch multiple tasks to reduce atomics
    return atomicAdd(counter, CHUNK);
}

// --- Vectorized Memory Access (Vec2) ---
// Enables loading/storing two elements at once for better memory throughput
template<typename T> struct Vec2;

template<> struct Vec2<half> {
    using V = half2;
    static __device__ __forceinline__ V load2(const half* p) { return *reinterpret_cast<const half2*>(p); }
    static __device__ __forceinline__ void store2(half* p, V v) { *reinterpret_cast<half2*>(p) = v; }
    static __device__ __forceinline__ void unpack(V v, float &a, float &b) { a = __half2float(v.x); b = __half2float(v.y); }
    static __device__ __forceinline__ V pack(float a, float b) { return __halves2half2(__float2half_rn(a), __float2half_rn(b)); }
};

template<> struct Vec2<__nv_bfloat16> {
    using V = __nv_bfloat162;
    static __device__ __forceinline__ V load2(const __nv_bfloat16* p) { return *reinterpret_cast<const __nv_bfloat162*>(p); }
    static __device__ __forceinline__ void store2(__nv_bfloat16* p, V v) { *reinterpret_cast<__nv_bfloat162*>(p) = v; }
    static __device__ __forceinline__ void unpack(V v, float &a, float &b) { a = __bfloat162float(v.x); b = __bfloat162float(v.y); }
    static __device__ __forceinline__ V pack(float a, float b) { return __halves2bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b)); }
};

template<> struct Vec2<float> {
    using V = float2;
    static __device__ __forceinline__ V load2(const float* p) { return *reinterpret_cast<const float2*>(p); }
    static __device__ __forceinline__ void store2(float* p, V v) { *reinterpret_cast<float2*>(p) = v; }
    static __device__ __forceinline__ void unpack(V v, float &a, float &b) { a = v.x; b = v.y; }
    static __device__ __forceinline__ V pack(float a, float b) { return make_float2(a, b); }
};

// Heuristic to pick block size based on group size (G)
static inline int pick_block(int G) {
    int nvec = G / 2;
    int b = 32;
    while ((b << 1) <= nvec) b <<= 1;
    return b;
}

// --- Parallel Reduction ---
// Computes the sum of floats across a thread block using shuffle and shared memory
__device__ __forceinline__ float block_reduce_sum_cg_lane0(float v) {
    cg::thread_block tb = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(tb);

    // Warp-level reduction
    float warp_sum = cg::reduce(tile, v, cg::plus<float>());

    __shared__ float warp_sums[8];
    int lane = tile.thread_rank();
    int warp = (int)(tb.thread_rank() / 32);

    // Store warp results to shared memory
    if (lane == 0) warp_sums[warp] = warp_sum;
    tb.sync();

    // Final reduction by the first warp
    float block_sum = 0.0f;
    if (warp == 0) {
        float x = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        block_sum = cg::reduce(tile, x, cg::plus<float>());
    }
    return block_sum;
}

// ---------------------------
// Forward Pass: Fake Quantization
// Logic: s = clamp(abs(scale), min_scale, max_scale) * sign(scale),
//        zp = clamp(round(zp), qmin, qmax)
//        y = (clamp(round(x/s + zp), qmin, qmax) - zp) * s
// ---------------------------
template<typename XType, typename YType, typename ScaleType, typename ZpType, bool HasZp>
__global__ void fake_quant_fwd_vec2(
    const XType* __restrict__ x,         // Input tensor [N_groups, G]
    YType* __restrict__ y,               // Output tensor [N_groups, G]
    const ScaleType* __restrict__ scale, // Per-group scale [N_groups]
    const ZpType* __restrict__ zp,       // Per-group zero_point [N_groups]
    int32_t qmin, int32_t qmax,          // Quantization range
    int G                                // Group size
) {
    int g = blockIdx.x; // Group index
    int tid = threadIdx.x;

    __shared__ float s_sh;
    __shared__ float inv_s_sh;
    __shared__ float zpf_sh;
    if (tid == 0) {
        float raw_s = to_f32(scale[g]);
        float s_local = clamp_abs_sign(raw_s, 1e-5f, 1e4f);
        s_sh = s_local;
        inv_s_sh = 1.0f / s_local;
        float z_local = 0.0f;
        if constexpr (HasZp) {
            float z_raw = to_f32(zp[g]);
            int32_t z_i = clamp_i32((int32_t)round_bankers(z_raw), qmin, qmax);
            z_local = (float)z_i;
        }
        zpf_sh = z_local;
    }
    __syncthreads();
    float s = s_sh;
    float inv_s = inv_s_sh;
    float zpf = zpf_sh;

    using VX = typename Vec2<XType>::V;
    using VY = typename Vec2<YType>::V;

    int base = g * G;
    int nvec = G / 2;
    
    // Process two elements at a time
    #pragma unroll
    for (int vi = tid; vi < nvec; vi += blockDim.x) {
        int idx = base + vi * 2;

        VX xv = Vec2<XType>::load2(reinterpret_cast<const XType*>(x + idx));
        float x0, x1;
        Vec2<XType>::unpack(xv, x0, x1);

        // --- Element 0 ---
        float u0 = x0 * inv_s + zpf; // Normalize
        int32_t q0 = clamp_i32((int32_t)round_bankers(u0), qmin, qmax); // Quantize
        float y0 = ((float)q0 - zpf) * s; // Dequantize

        // --- Element 1 ---
        float u1 = x1 * inv_s + zpf;
        int32_t q1 = clamp_i32((int32_t)round_bankers(u1), qmin, qmax);
        float y1 = ((float)q1 - zpf) * s;

        VY yv = Vec2<YType>::pack(y0, y1);
        Vec2<YType>::store2(reinterpret_cast<YType*>(y + idx), yv);
    }
}

// ---------------------------
// Forward Pass: Persistent Fake Quantization
// ---------------------------
template<typename XType, typename YType, typename ScaleType, typename ZpType, bool HasZp>
__global__ void fake_quant_fwd_vec2_persistent(
    const XType* __restrict__ x,         // Input tensor [N_groups, G]
    YType* __restrict__ y,               // Output tensor [N_groups, G]
    const ScaleType* __restrict__ scale, // Per-group scale [N_groups]
    const ZpType* __restrict__ zp,       // Per-group zero_point [N_groups]
    int32_t qmin, int32_t qmax,          // Quantization range
    int G,                               // Group size
    int num_tasks,                       // Number of groups
    int* __restrict__ counter            // Persistent task counter
) {
    int tid = threadIdx.x;

    __shared__ int g_sh;
    __shared__ float s_sh;
    __shared__ float inv_s_sh;
    __shared__ float zpf_sh;

    while (true) {
        if (tid == 0) g_sh = persistent_next(counter);
        __syncthreads();
        int g = g_sh;
        if (g >= num_tasks) break;

        if (tid == 0) {
            float raw_s = to_f32(scale[g]);
            float s_local = clamp_abs_sign(raw_s, 1e-5f, 1e4f);
            s_sh = s_local;
            inv_s_sh = 1.0f / s_local;
            float z_local = 0.0f;
            if constexpr (HasZp) {
                float z_raw = to_f32(zp[g]);
                int32_t z_i = clamp_i32((int32_t)round_bankers(z_raw), qmin, qmax);
                z_local = (float)z_i;
            }
            zpf_sh = z_local;
        }
        __syncthreads();

        float s = s_sh;
        float inv_s = inv_s_sh;
        float zpf = zpf_sh;

        using VX = typename Vec2<XType>::V;
        using VY = typename Vec2<YType>::V;

        int base = g * G;
        int nvec = G / 2;

        // Process two elements at a time
        #pragma unroll
        for (int vi = tid; vi < nvec; vi += blockDim.x) {
            int idx = base + vi * 2;

            VX xv = Vec2<XType>::load2(reinterpret_cast<const XType*>(x + idx));
            float x0, x1;
            Vec2<XType>::unpack(xv, x0, x1);

            // --- Element 0 ---
            float u0 = x0 * inv_s + zpf; // Normalize
            int32_t q0 = clamp_i32((int32_t)round_bankers(u0), qmin, qmax); // Quantize
            float y0 = ((float)q0 - zpf) * s; // Dequantize

            // --- Element 1 ---
            float u1 = x1 * inv_s + zpf;
            int32_t q1 = clamp_i32((int32_t)round_bankers(u1), qmin, qmax);
            float y1 = ((float)q1 - zpf) * s;

            VY yv = Vec2<YType>::pack(y0, y1);
            Vec2<YType>::store2(reinterpret_cast<YType*>(y + idx), yv);
        }

        // Ensure all threads finish before shared is rewritten next loop
        __syncthreads();
    }
}

// ---------------------------
// Backward Pass: Gradient Calculation
// Logic using Straight-Through Estimator (STE):
// 1. dx     = dy * mask (mask = 1 if rounded pre-clamp is within [qmin, qmax], else 0)
// 2. dscale = dy * [(q - zp) - mask * (x/s)]
// 3. dzp    = dy * s * (mask - 1)
// ---------------------------
template<typename XType, typename DyType, typename DxType, typename ScaleType, typename ZpType, bool HasZp>
__global__ void fake_quant_bwd_vec2(
    const XType* __restrict__ x,         // Original input
    const DyType* __restrict__ dy,       // Gradient of output
    DxType* __restrict__ dx,             // Gradient of input
    const ScaleType* __restrict__ scale, // Scales
    const ZpType* __restrict__ zp,       // Zero points
    ScaleType* __restrict__ dscale,      // Output gradient of scale
    ZpType* __restrict__ dzp,            // Output gradient of zero_point
    int32_t qmin, int32_t qmax,
    int G
) {
    cg::thread_block tb = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(tb);

    int g = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tile.thread_rank();
    int warp = (int)(tb.thread_rank() / 32);

    __shared__ float s_sh;
    __shared__ float inv_s_sh;
    __shared__ float zpf_sh;
    __shared__ float spg_sh;
    if (tid == 0) {
        float raw_s = to_f32(scale[g]);
        float s_local = clamp_abs_sign(raw_s, 1e-5f, 1e4f);
        s_sh = s_local;
        inv_s_sh = 1.0f / s_local;
        spg_sh = 1.0f;
        float z_local = 0.0f;
        if constexpr (HasZp) {
            float z_raw = to_f32(zp[g]);
            int32_t z_i = clamp_i32((int32_t)round_bankers(z_raw), qmin, qmax);
            z_local = (float)z_i;
        }
        zpf_sh = z_local;
    }
    tb.sync();
    float s = s_sh;
    float inv_s = inv_s_sh;
    float zpf = zpf_sh;
    float spg = spg_sh;

    using VX  = typename Vec2<XType>::V;
    using VDY = typename Vec2<DyType>::V;
    using VDX = typename Vec2<DxType>::V;

    int base = g * G;
    int nvec = G / 2;

    float local_dscale = 0.0f;
    float local_dzp = 0.0f;
    
    #pragma unroll
    for (int vi = tid; vi < nvec; vi += blockDim.x) {
        int idx = base + vi * 2;

        VX  xv  = Vec2<XType>::load2(reinterpret_cast<const XType*>(x + idx));
        VDY dyv = Vec2<DyType>::load2(reinterpret_cast<const DyType*>(dy + idx));

        float x0, x1, g0, g1;
        Vec2<XType>::unpack(xv, x0, x1);
        Vec2<DyType>::unpack(dyv, g0, g1);

        // --- Element 0 ---
        float u0 = x0 * inv_s + zpf;
        // Match PyTorch STE: clamp mask is based on rounded (pre-clamp) value.
        int32_t q0_unclamped = (int32_t)round_bankers(u0);
        float m0 = (q0_unclamped >= qmin && q0_unclamped <= qmax) ? 1.0f : 0.0f;
        int32_t q0 = clamp_i32(q0_unclamped, qmin, qmax);
        
        float dx0 = g0 * m0; // Input gradient
        // Scale gradient formula: dy * ((q - zp) - mask * (x/s))
        local_dscale += g0 * ( ((float)q0 - zpf) - m0 * (x0 * inv_s) );
        // Zero point gradient formula: dy * s * (mask - 1)
        if constexpr (HasZp) local_dzp += g0 * s * (m0 - 1.0f);

        // --- Element 1 ---
        float u1 = x1 * inv_s + zpf;
        int32_t q1_unclamped = (int32_t)round_bankers(u1);
        float m1 = (q1_unclamped >= qmin && q1_unclamped <= qmax) ? 1.0f : 0.0f;
        int32_t q1 = clamp_i32(q1_unclamped, qmin, qmax);
        
        float dx1 = g1 * m1;
        local_dscale += g1 * ( ((float)q1 - zpf) - m1 * (x1 * inv_s) );
        if constexpr (HasZp) local_dzp += g1 * s * (m1 - 1.0f);

        VDX dxv = Vec2<DxType>::pack(dx0, dx1);
        Vec2<DxType>::store2(reinterpret_cast<DxType*>(dx + idx), dxv);
    }

    // Accumulate dscale and dzp gradients across the block
    float sum_dscale = block_reduce_sum_cg_lane0(local_dscale);
    float sum_dzp = 0.0f;
    if constexpr (HasZp) sum_dzp = block_reduce_sum_cg_lane0(local_dzp);

    // Final write to global memory by a single thread per group
    if (warp == 0 && lane == 0) {
        float ds_raw = sum_dscale * spg;
        dscale[g] = from_f32<ScaleType>(ds_raw);
        if constexpr (HasZp) dzp[g] = from_f32<ZpType>(sum_dzp);
    }
}

// ---------------------------
// Backward Pass: Persistent Gradient Calculation
// ---------------------------
template<typename XType, typename DyType, typename DxType, typename ScaleType, typename ZpType, bool HasZp>
__global__ void fake_quant_bwd_vec2_persistent(
    const XType* __restrict__ x,         // Original input
    const DyType* __restrict__ dy,       // Gradient of output
    DxType* __restrict__ dx,             // Gradient of input
    const ScaleType* __restrict__ scale, // Scales
    const ZpType* __restrict__ zp,       // Zero points
    ScaleType* __restrict__ dscale,      // Output gradient of scale
    ZpType* __restrict__ dzp,            // Output gradient of zero_point
    int32_t qmin, int32_t qmax,
    int G,
    int num_tasks,
    int* __restrict__ counter
) {
    cg::thread_block tb = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(tb);

    int tid = threadIdx.x;
    int lane = tile.thread_rank();
    int warp = (int)(tb.thread_rank() / 32);

    __shared__ int g_sh;
    __shared__ float s_sh;
    __shared__ float inv_s_sh;
    __shared__ float zpf_sh;
    __shared__ float spg_sh;

    while (true) {
        if (tid == 0) g_sh = persistent_next(counter);
        tb.sync();
        int g = g_sh;
        if (g >= num_tasks) break;

        if (tid == 0) {
            float raw_s = to_f32(scale[g]);
            float s_local = clamp_abs_sign(raw_s, 1e-5f, 1e4f);
            s_sh = s_local;
            inv_s_sh = 1.0f / s_local;
            spg_sh = 1.0f;
            float z_local = 0.0f;
            if constexpr (HasZp) {
                float z_raw = to_f32(zp[g]);
                int32_t z_i = clamp_i32((int32_t)round_bankers(z_raw), qmin, qmax);
                z_local = (float)z_i;
            }
            zpf_sh = z_local;
        }
        tb.sync();

        float s = s_sh;
        float inv_s = inv_s_sh;
        float zpf = zpf_sh;
        float spg = spg_sh;

        using VX  = typename Vec2<XType>::V;
        using VDY = typename Vec2<DyType>::V;
        using VDX = typename Vec2<DxType>::V;

        int base = g * G;
        int nvec = G / 2;

        float local_dscale = 0.0f;
        float local_dzp = 0.0f;

        #pragma unroll
        for (int vi = tid; vi < nvec; vi += blockDim.x) {
            int idx = base + vi * 2;

            VX  xv  = Vec2<XType>::load2(reinterpret_cast<const XType*>(x + idx));
            VDY dyv = Vec2<DyType>::load2(reinterpret_cast<const DyType*>(dy + idx));

            float x0, x1, g0, g1;
            Vec2<XType>::unpack(xv, x0, x1);
            Vec2<DyType>::unpack(dyv, g0, g1);

            // --- Element 0 ---
            float u0 = x0 * inv_s + zpf;
            // Match PyTorch STE: clamp mask is based on rounded (pre-clamp) value.
            int32_t q0_unclamped = (int32_t)round_bankers(u0);
            float m0 = (q0_unclamped >= qmin && q0_unclamped <= qmax) ? 1.0f : 0.0f;
            int32_t q0 = clamp_i32(q0_unclamped, qmin, qmax);

            float dx0 = g0 * m0; // Input gradient
            // Scale gradient formula: dy * ((q - zp) - mask * (x/s))
            local_dscale += g0 * ( ((float)q0 - zpf) - m0 * (x0 * inv_s) );
            // Zero point gradient formula: dy * s * (mask - 1)
            if constexpr (HasZp) local_dzp += g0 * s * (m0 - 1.0f);

            // --- Element 1 ---
            float u1 = x1 * inv_s + zpf;
            int32_t q1_unclamped = (int32_t)round_bankers(u1);
            float m1 = (q1_unclamped >= qmin && q1_unclamped <= qmax) ? 1.0f : 0.0f;
            int32_t q1 = clamp_i32(q1_unclamped, qmin, qmax);

            float dx1 = g1 * m1;
            local_dscale += g1 * ( ((float)q1 - zpf) - m1 * (x1 * inv_s) );
            if constexpr (HasZp) local_dzp += g1 * s * (m1 - 1.0f);

            VDX dxv = Vec2<DxType>::pack(dx0, dx1);
            Vec2<DxType>::store2(reinterpret_cast<DxType*>(dx + idx), dxv);
        }

        // Accumulate dscale and dzp gradients across the block
        float sum_dscale = block_reduce_sum_cg_lane0(local_dscale);
        float sum_dzp = 0.0f;
        if constexpr (HasZp) sum_dzp = block_reduce_sum_cg_lane0(local_dzp);

        // Final write to global memory by a single thread per group
        if (warp == 0 && lane == 0) {
            float ds_raw = sum_dscale * spg;
            dscale[g] = from_f32<ScaleType>(ds_raw);
            if constexpr (HasZp) dzp[g] = from_f32<ZpType>(sum_dzp);
        }

        // Ensure all threads finish before shared is rewritten next loop
        tb.sync();
    }
}


// ---------------------------
// Template Instantations
// ---------------------------
#define INST_ONE(XT, ST, ZT, HZ) \
  template __global__ void fake_quant_fwd_vec2<XT, XT, ST, ZT, HZ>( \
      const XT*, XT*, const ST*, const ZT*, int32_t, int32_t, int); \
  template __global__ void fake_quant_fwd_vec2_persistent<XT, XT, ST, ZT, HZ>( \
      const XT*, XT*, const ST*, const ZT*, int32_t, int32_t, int, int, int*); \
  template __global__ void fake_quant_bwd_vec2<XT, XT, XT, ST, ZT, HZ>( \
      const XT*, const XT*, XT*, const ST*, const ZT*, ST*, ZT*, int32_t, int32_t, int); \
  template __global__ void fake_quant_bwd_vec2_persistent<XT, XT, XT, ST, ZT, HZ>( \
      const XT*, const XT*, XT*, const ST*, const ZT*, ST*, ZT*, int32_t, int32_t, int, int, int*);

#define INST_SCALE_ZP(XT, ST) \
  INST_ONE(XT, ST, float, false) \
  INST_ONE(XT, ST, float, true)  \
  INST_ONE(XT, ST, half, true)   \
  INST_ONE(XT, ST, __nv_bfloat16, true)

#define INST_SCALE(XT) \
  INST_SCALE_ZP(XT, half) \
  INST_SCALE_ZP(XT, __nv_bfloat16) \
  INST_SCALE_ZP(XT, float)

INST_SCALE(half)
INST_SCALE(__nv_bfloat16)
INST_SCALE(float)

#undef INST_SCALE
#undef INST_SCALE_ZP
#undef INST_ONE

// ---------------------------
// Python Wrapper: Forward
// ---------------------------
torch::Tensor fake_quant_fwd_cuda(
    torch::Tensor x, torch::Tensor scale, c10::optional<torch::Tensor> zp_opt,
    int64_t qmin, int64_t qmax, int64_t group_size
) {
    CHECK_CUDA(x); CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(scale);

    int G = (int)group_size;
    CHECK_SHAPE((G == 64 || G == 128 || G == 256), "group_size must be 64/128/256");
    int64_t N = x.numel();
    CHECK_SHAPE(N % G == 0, "x.numel must be divisible by group_size");
    int N_groups = (int)(N / G);
    CHECK_SHAPE(scale.numel() == N_groups, "scale numel must equal N_groups");

    bool has_zp = zp_opt.has_value();
    torch::Tensor zp;
    auto zp_st = torch::ScalarType::Undefined;
    if (has_zp) {
        zp = zp_opt.value();
        CHECK_CUDA(zp); CHECK_CONTIGUOUS(zp);
        TORCH_CHECK(
            zp.scalar_type() == torch::kFloat16 ||
            zp.scalar_type() == torch::kBFloat16 ||
            zp.scalar_type() == torch::kFloat32,
            "zp dtype must be fp16/bf16/fp32"
        );
        CHECK_SHAPE(zp.numel() == N_groups, "zp numel must equal N_groups");
        zp_st = zp.scalar_type();
    }

    auto y = torch::empty_like(x);
    const at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    auto counter = torch::empty({1}, x.options().dtype(torch::kInt32));
    cudaMemsetAsync(counter.data_ptr<int>(), 0, sizeof(int), stream);

    auto props = at::cuda::getCurrentDeviceProperties();
    int num_sms = props->multiProcessorCount;
    int k = 2;
    int resident_ctas = num_sms * k;

    dim3 grid(resident_ctas);
    dim3 block(pick_block(G));

    auto x_st = x.scalar_type();
    auto s_st = scale.scalar_type();

    // Dispatch based on dtypes and whether ZP exists
    auto dispatch_scale = [&](auto XTag, auto STag) {
        using XType = typename decltype(XTag)::type;
        using ScaleType = typename decltype(STag)::type;
        const XType* x_ptr = reinterpret_cast<const XType*>(x.data_ptr());
        XType* y_ptr = reinterpret_cast<XType*>(y.data_ptr());
        const ScaleType* s_ptr = reinterpret_cast<const ScaleType*>(scale.data_ptr());

        if (has_zp) {
            auto dispatch_zp = [&](auto ZTag) {
                using ZpType = typename decltype(ZTag)::type;
                const ZpType* zp_ptr = reinterpret_cast<const ZpType*>(zp.data_ptr());
                fake_quant_fwd_vec2_persistent<XType, XType, ScaleType, ZpType, true>
                    <<<grid, block, 0, stream>>>(x_ptr, y_ptr, s_ptr, zp_ptr,
                                                 (int32_t)qmin, (int32_t)qmax, G,
                                                 N_groups, counter.data_ptr<int>());
            };
            if (zp_st == torch::kFloat16) dispatch_zp(TypeTag<half>{});
            else if (zp_st == torch::kBFloat16) dispatch_zp(TypeTag<__nv_bfloat16>{});
            else if (zp_st == torch::kFloat32) dispatch_zp(TypeTag<float>{});
            else TORCH_CHECK(false, "zp dtype must be fp16/bf16/fp32");
        } else {
            fake_quant_fwd_vec2_persistent<XType, XType, ScaleType, float, false>
                <<<grid, block, 0, stream>>>(x_ptr, y_ptr, s_ptr, nullptr,
                                             (int32_t)qmin, (int32_t)qmax, G,
                                             N_groups, counter.data_ptr<int>());
        }
    };

    auto dispatch_x = [&](auto XTag) {
        if (s_st == torch::kFloat16) dispatch_scale(XTag, TypeTag<half>{});
        else if (s_st == torch::kBFloat16) dispatch_scale(XTag, TypeTag<__nv_bfloat16>{});
        else if (s_st == torch::kFloat32) dispatch_scale(XTag, TypeTag<float>{});
        else TORCH_CHECK(false, "scale dtype must be fp16/bf16/fp32");
    };

    if (x_st == torch::kFloat16) dispatch_x(TypeTag<half>{});
    else if (x_st == torch::kBFloat16) dispatch_x(TypeTag<__nv_bfloat16>{});
    else if (x_st == torch::kFloat32) dispatch_x(TypeTag<float>{});
    else TORCH_CHECK(false, "x dtype must be fp16/bf16/fp32");

    return y;
}

// ---------------------------
// Python Wrapper: Backward
// ---------------------------
std::vector<torch::Tensor> fake_quant_bwd_cuda(
    torch::Tensor x, torch::Tensor dy, torch::Tensor scale, c10::optional<torch::Tensor> zp_opt,
    int64_t qmin, int64_t qmax, int64_t group_size
) {
    CHECK_CUDA(x); CHECK_CUDA(dy); CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(dy); CHECK_CONTIGUOUS(scale);

    CHECK_SHAPE(dy.sizes() == x.sizes(), "dy must have same shape as x");
    int G = (int)group_size;
    CHECK_SHAPE((G == 64 || G == 128 || G == 256), "group_size must be 64/128/256");
    int64_t N = x.numel();
    CHECK_SHAPE(N % G == 0, "x.numel must be divisible by group_size");
    int N_groups = (int)(N / G);
    CHECK_SHAPE(scale.numel() == N_groups, "scale numel must equal N_groups");

    bool has_zp = zp_opt.has_value();
    torch::Tensor zp;
    auto zp_st = torch::ScalarType::Undefined;
    if (has_zp) {
        zp = zp_opt.value();
        CHECK_CUDA(zp); CHECK_CONTIGUOUS(zp);
        TORCH_CHECK(
            zp.scalar_type() == torch::kFloat16 ||
            zp.scalar_type() == torch::kBFloat16 ||
            zp.scalar_type() == torch::kFloat32,
            "zp dtype must be fp16/bf16/fp32"
        );
        CHECK_SHAPE(zp.numel() == N_groups, "zp numel must equal N_groups");
        zp_st = zp.scalar_type();
    }

    auto dx = torch::empty_like(x);
    auto dscale = torch::empty_like(scale);
    torch::Tensor dzp;
    if (has_zp) {
        dzp = torch::empty_like(zp);
    } else {
        dzp = torch::empty({0}, x.options().dtype(torch::kFloat32)); // placeholder
    }

    const at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    auto counter = torch::empty({1}, x.options().dtype(torch::kInt32));
    cudaMemsetAsync(counter.data_ptr<int>(), 0, sizeof(int), stream);

    auto props = at::cuda::getCurrentDeviceProperties();
    int num_sms = props->multiProcessorCount;
    int k = 2;
    int resident_ctas = num_sms * k;

    dim3 grid(resident_ctas);
    dim3 block(pick_block(G));

    auto x_st = x.scalar_type();
    auto s_st = scale.scalar_type();

    auto dispatch_scale = [&](auto XTag, auto STag) {
        using XType = typename decltype(XTag)::type;
        using ScaleType = typename decltype(STag)::type;

        const XType* x_ptr = reinterpret_cast<const XType*>(x.data_ptr());
        const XType* dy_ptr = reinterpret_cast<const XType*>(dy.data_ptr());
        XType* dx_ptr = reinterpret_cast<XType*>(dx.data_ptr());
        const ScaleType* s_ptr = reinterpret_cast<const ScaleType*>(scale.data_ptr());
        ScaleType* ds_ptr = reinterpret_cast<ScaleType*>(dscale.data_ptr());
        if (has_zp) {
            auto dispatch_zp = [&](auto ZTag) {
                using ZpType = typename decltype(ZTag)::type;
                const ZpType* zp_ptr = reinterpret_cast<const ZpType*>(zp.data_ptr());
                ZpType* dzp_ptr = reinterpret_cast<ZpType*>(dzp.data_ptr());
                fake_quant_bwd_vec2_persistent<XType, XType, XType, ScaleType, ZpType, true>
                    <<<grid, block, 0, stream>>>(x_ptr, dy_ptr, dx_ptr, s_ptr, zp_ptr,
                                                 ds_ptr, dzp_ptr,
                                                 (int32_t)qmin, (int32_t)qmax, G,
                                                 N_groups, counter.data_ptr<int>());
            };
            if (zp_st == torch::kFloat16) dispatch_zp(TypeTag<half>{});
            else if (zp_st == torch::kBFloat16) dispatch_zp(TypeTag<__nv_bfloat16>{});
            else if (zp_st == torch::kFloat32) dispatch_zp(TypeTag<float>{});
            else TORCH_CHECK(false, "zp dtype must be fp16/bf16/fp32");
        } else {
            fake_quant_bwd_vec2_persistent<XType, XType, XType, ScaleType, float, false>
                <<<grid, block, 0, stream>>>(x_ptr, dy_ptr, dx_ptr, s_ptr, nullptr,
                                             ds_ptr, nullptr,
                                             (int32_t)qmin, (int32_t)qmax, G,
                                             N_groups, counter.data_ptr<int>());
        }
    };

    auto dispatch_x = [&](auto XTag) {
        if (s_st == torch::kFloat16) dispatch_scale(XTag, TypeTag<half>{});
        else if (s_st == torch::kBFloat16) dispatch_scale(XTag, TypeTag<__nv_bfloat16>{});
        else if (s_st == torch::kFloat32) dispatch_scale(XTag, TypeTag<float>{});
        else TORCH_CHECK(false, "scale dtype must be fp16/bf16/fp32");
    };

    if (x_st == torch::kFloat16) dispatch_x(TypeTag<half>{});
    else if (x_st == torch::kBFloat16) dispatch_x(TypeTag<__nv_bfloat16>{});
    else if (x_st == torch::kFloat32) dispatch_x(TypeTag<float>{});
    else TORCH_CHECK(false, "x dtype must be fp16/bf16/fp32");

    return {dx, dscale, dzp};
}

// ---------------------------
// Seq2Bit specialized kernels
// Levels: {-0.75, -0.25, 0.25, 0.75} * alpha
// code = clamp(round((clamp(x/alpha, -1, 1) + 0.75) / 0.5), 0, 3)
// ---------------------------
template<typename XType, typename AType>
__global__ void fake_quant_seq2bit_fwd_vec2(
    const XType* __restrict__ x,
    XType* __restrict__ y,
    const AType* __restrict__ alpha,
    int G
) {
    int g = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float s_sh;
    __shared__ float inv_s_sh;
    if (tid == 0) {
        float raw_a = to_f32(alpha[g]);
        float s_local = clamp_abs_sign(raw_a, 1e-6f, 1e4f);
        s_sh = s_local;
        inv_s_sh = 1.0f / s_local;
    }
    __syncthreads();

    float s = s_sh;
    float inv_s = inv_s_sh;

    using VX = typename Vec2<XType>::V;
    int base = g * G;
    int nvec = G / 2;
    for (int vi = tid; vi < nvec; vi += blockDim.x) {
        int idx = base + vi * 2;
        VX xv = Vec2<XType>::load2(reinterpret_cast<const XType*>(x + idx));
        float x0, x1;
        Vec2<XType>::unpack(xv, x0, x1);

        float v0 = x0 * inv_s;
        float xn0 = fminf(fmaxf(v0, -1.0f), 1.0f);
        float u0 = (xn0 + 0.75f) * 2.0f;
        int32_t q0 = clamp_i32((int32_t)round_bankers(u0), 0, 3);
        float y0 = (((float)q0) * 0.5f - 0.75f) * s;

        float v1 = x1 * inv_s;
        float xn1 = fminf(fmaxf(v1, -1.0f), 1.0f);
        float u1 = (xn1 + 0.75f) * 2.0f;
        int32_t q1 = clamp_i32((int32_t)round_bankers(u1), 0, 3);
        float y1 = (((float)q1) * 0.5f - 0.75f) * s;

        using VY = typename Vec2<XType>::V;
        VY yv = Vec2<XType>::pack(y0, y1);
        Vec2<XType>::store2(reinterpret_cast<XType*>(y + idx), yv);
    }
}

template<typename XType, typename AType>
__global__ void fake_quant_seq2bit_bwd_vec2(
    const XType* __restrict__ x,
    const XType* __restrict__ dy,
    XType* __restrict__ dx,
    const AType* __restrict__ alpha,
    AType* __restrict__ dalpha,
    int G
) {
    cg::thread_block tb = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(tb);
    int tid = threadIdx.x;
    int lane = tile.thread_rank();
    int warp = (int)(tb.thread_rank() / 32);
    int g = blockIdx.x;

    __shared__ float s_sh;
    __shared__ float inv_s_sh;
    if (tid == 0) {
        float raw_a = to_f32(alpha[g]);
        float s_local = clamp_abs_sign(raw_a, 1e-6f, 1e4f);
        s_sh = s_local;
        inv_s_sh = 1.0f / s_local;
    }
    tb.sync();

    float s = s_sh;
    float inv_s = inv_s_sh;

    using VX = typename Vec2<XType>::V;
    using VDY = typename Vec2<XType>::V;
    using VDX = typename Vec2<XType>::V;

    int base = g * G;
    int nvec = G / 2;
    float local_da = 0.0f;

    for (int vi = tid; vi < nvec; vi += blockDim.x) {
        int idx = base + vi * 2;
        VX xv = Vec2<XType>::load2(reinterpret_cast<const XType*>(x + idx));
        VDY dyv = Vec2<XType>::load2(reinterpret_cast<const XType*>(dy + idx));
        float x0, x1, gy0, gy1;
        Vec2<XType>::unpack(xv, x0, x1);
        Vec2<XType>::unpack(dyv, gy0, gy1);

        float v0 = x0 * inv_s;
        float mx0 = (v0 >= -1.0f && v0 <= 1.0f) ? 1.0f : 0.0f;
        float xn0 = fminf(fmaxf(v0, -1.0f), 1.0f);
        float u0 = (xn0 + 0.75f) * 2.0f;
        int32_t q0_un = (int32_t)round_bankers(u0);
        float mq0 = (q0_un >= 0 && q0_un <= 3) ? 1.0f : 0.0f;
        float m0 = mx0 * mq0;
        int32_t q0 = clamp_i32(q0_un, 0, 3);
        float lv0 = ((float)q0) * 0.5f - 0.75f;
        float dx0 = gy0 * m0;
        local_da += gy0 * (lv0 - m0 * (x0 * inv_s));

        float v1 = x1 * inv_s;
        float mx1 = (v1 >= -1.0f && v1 <= 1.0f) ? 1.0f : 0.0f;
        float xn1 = fminf(fmaxf(v1, -1.0f), 1.0f);
        float u1 = (xn1 + 0.75f) * 2.0f;
        int32_t q1_un = (int32_t)round_bankers(u1);
        float mq1 = (q1_un >= 0 && q1_un <= 3) ? 1.0f : 0.0f;
        float m1 = mx1 * mq1;
        int32_t q1 = clamp_i32(q1_un, 0, 3);
        float lv1 = ((float)q1) * 0.5f - 0.75f;
        float dx1 = gy1 * m1;
        local_da += gy1 * (lv1 - m1 * (x1 * inv_s));

        VDX dxv = Vec2<XType>::pack(dx0, dx1);
        Vec2<XType>::store2(reinterpret_cast<XType*>(dx + idx), dxv);
    }

    float sum_da = block_reduce_sum_cg_lane0(local_da);
    if (warp == 0 && lane == 0) {
        dalpha[g] = from_f32<AType>(sum_da);
    }
}

torch::Tensor fake_quant_ste_seq2bit_fwd_cuda(torch::Tensor x, torch::Tensor alpha, int64_t group_size) {
    CHECK_CUDA(x); CHECK_CUDA(alpha);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(alpha);
    int G = (int)group_size;
    CHECK_SHAPE((G == 64 || G == 128 || G == 256), "group_size must be 64/128/256");
    int64_t N = x.numel();
    CHECK_SHAPE(N % G == 0, "x.numel must be divisible by group_size");
    int N_groups = (int)(N / G);
    CHECK_SHAPE(alpha.numel() == N_groups, "alpha numel must equal N_groups");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32,
        "x dtype must be fp16/bf16/fp32"
    );
    TORCH_CHECK(
        alpha.scalar_type() == torch::kFloat16 || alpha.scalar_type() == torch::kBFloat16 || alpha.scalar_type() == torch::kFloat32,
        "alpha dtype must be fp16/bf16/fp32"
    );

    auto y = torch::empty_like(x);
    const at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    dim3 grid(N_groups);
    dim3 block(pick_block(G));

    auto x_st = x.scalar_type();
    auto a_st = alpha.scalar_type();
    auto dispatch_a = [&](auto XTag, auto ATag) {
        using XType = typename decltype(XTag)::type;
        using AType = typename decltype(ATag)::type;
        fake_quant_seq2bit_fwd_vec2<XType, AType><<<grid, block, 0, stream>>>(
            reinterpret_cast<const XType*>(x.data_ptr()),
            reinterpret_cast<XType*>(y.data_ptr()),
            reinterpret_cast<const AType*>(alpha.data_ptr()),
            G
        );
    };
    auto dispatch_x = [&](auto XTag) {
        if (a_st == torch::kFloat16) dispatch_a(XTag, TypeTag<half>{});
        else if (a_st == torch::kBFloat16) dispatch_a(XTag, TypeTag<__nv_bfloat16>{});
        else if (a_st == torch::kFloat32) dispatch_a(XTag, TypeTag<float>{});
        else TORCH_CHECK(false, "alpha dtype must be fp16/bf16/fp32");
    };
    if (x_st == torch::kFloat16) dispatch_x(TypeTag<half>{});
    else if (x_st == torch::kBFloat16) dispatch_x(TypeTag<__nv_bfloat16>{});
    else if (x_st == torch::kFloat32) dispatch_x(TypeTag<float>{});
    else TORCH_CHECK(false, "x dtype must be fp16/bf16/fp32");
    return y;
}

std::vector<torch::Tensor> fake_quant_ste_seq2bit_bwd_cuda(
    torch::Tensor x, torch::Tensor dy, torch::Tensor alpha, int64_t group_size
) {
    CHECK_CUDA(x); CHECK_CUDA(dy); CHECK_CUDA(alpha);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(dy); CHECK_CONTIGUOUS(alpha);
    CHECK_SHAPE(dy.sizes() == x.sizes(), "dy must have same shape as x");
    int G = (int)group_size;
    CHECK_SHAPE((G == 64 || G == 128 || G == 256), "group_size must be 64/128/256");
    int64_t N = x.numel();
    CHECK_SHAPE(N % G == 0, "x.numel must be divisible by group_size");
    int N_groups = (int)(N / G);
    CHECK_SHAPE(alpha.numel() == N_groups, "alpha numel must equal N_groups");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32,
        "x dtype must be fp16/bf16/fp32"
    );
    TORCH_CHECK(
        alpha.scalar_type() == torch::kFloat16 || alpha.scalar_type() == torch::kBFloat16 || alpha.scalar_type() == torch::kFloat32,
        "alpha dtype must be fp16/bf16/fp32"
    );

    auto dx = torch::empty_like(x);
    auto dalpha = torch::empty_like(alpha);
    const at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    dim3 grid(N_groups);
    dim3 block(pick_block(G));

    auto x_st = x.scalar_type();
    auto a_st = alpha.scalar_type();
    auto dispatch_a = [&](auto XTag, auto ATag) {
        using XType = typename decltype(XTag)::type;
        using AType = typename decltype(ATag)::type;
        fake_quant_seq2bit_bwd_vec2<XType, AType><<<grid, block, 0, stream>>>(
            reinterpret_cast<const XType*>(x.data_ptr()),
            reinterpret_cast<const XType*>(dy.data_ptr()),
            reinterpret_cast<XType*>(dx.data_ptr()),
            reinterpret_cast<const AType*>(alpha.data_ptr()),
            reinterpret_cast<AType*>(dalpha.data_ptr()),
            G
        );
    };
    auto dispatch_x = [&](auto XTag) {
        if (a_st == torch::kFloat16) dispatch_a(XTag, TypeTag<half>{});
        else if (a_st == torch::kBFloat16) dispatch_a(XTag, TypeTag<__nv_bfloat16>{});
        else if (a_st == torch::kFloat32) dispatch_a(XTag, TypeTag<float>{});
        else TORCH_CHECK(false, "alpha dtype must be fp16/bf16/fp32");
    };
    if (x_st == torch::kFloat16) dispatch_x(TypeTag<half>{});
    else if (x_st == torch::kBFloat16) dispatch_x(TypeTag<__nv_bfloat16>{});
    else if (x_st == torch::kFloat32) dispatch_x(TypeTag<float>{});
    else TORCH_CHECK(false, "x dtype must be fp16/bf16/fp32");
    return {dx, dalpha};
}

// --- Bindings ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fake_quant_fwd_cuda, "FakeQuant forward (CUDA)");
    m.def("bwd", &fake_quant_bwd_cuda, "FakeQuant backward (CUDA): returns dx, dscale, dzp(or empty)");
    m.def(
        "fake_quant_ste_seq2bit_fwd_cuda",
        &fake_quant_ste_seq2bit_fwd_cuda,
        "Seq2Bit FakeQuant forward (CUDA): returns y"
    );
    m.def(
        "fake_quant_ste_seq2bit_bwd_cuda",
        &fake_quant_ste_seq2bit_bwd_cuda,
        "Seq2Bit FakeQuant backward (CUDA): returns dx, dalpha"
    );
}
