
# EfficientQAT FakeQuant CUDA Kernel 优化报告（带代码锚点与案例）

## 0. 这份 kernel 实际做的事情（先统一"执行语义"）

你的实现把 FakeQuant 的**前向**和**反向**都做成了"**按 group（块）处理**"的 kernel：

* 输入 `x` 是一维连续张量，但逻辑上按 `[N_groups, G]` 解释（`G=64/128/256`）
* `scale[g]`（以及可选的 `zp[g]`）是 **per-group 常量**
* 每个 group 内做逐元素 fake-quant：
  [
  y = ( \mathrm{clamp}(\mathrm{round}(x/s+zp), qmin,qmax)-zp )\cdot s
  ]

反向是 STE：

* `dx = dy * mask`
* `dscale`/`dzp` 在 group 内求和归约

> 关键点：你的 Python wrapper **默认总是走 persistent 版本**（`*_persistent`），非 persistent 的 kernel 主要是备用/对照用。

---

## 1. 优化点 #1：Persistent Kernel（常驻线程块）——解决"launch 爆炸"和"尾效应"

### 对应代码锚点

* persistent work queue：
  `persistent_next(counter) -> atomicAdd(counter, 1)`
* persistent 前向/反向 kernel：
  `fake_quant_fwd_vec2_persistent(...)` / `fake_quant_bwd_vec2_persistent(...)`
* wrapper 里固定 resident blocks：

  ```cpp
  int resident_ctas = num_sms * k; // k=2
  dim3 grid(resident_ctas);
  cudaMemsetAsync(counter, 0, sizeof(int), stream);
  ```

### Persistent Kernel 代码编写范式

#### 范式1：全局任务队列与计数器初始化

```cpp
// 1. 在 Python wrapper 中创建并初始化任务计数器
auto counter = torch::empty({1}, x.options().dtype(torch::kInt32));
cudaMemsetAsync(counter.data_ptr<int>(), 0, sizeof(int), stream);
```

#### 范式2：计算并启动常驻线程块

```cpp
// 2. 根据设备 SM 数量计算常驻线程块数
auto props = at::cuda::getCurrentDeviceProperties();
int num_sms = props->multiProcessorCount;
int k = 2;  // 每个 SM 预留的常驻线程块数
int resident_ctas = num_sms * k;

// 3. 启动固定数量的线程块，远小于任务总数
dim3 grid(resident_ctas);
dim3 block(pick_block(G));  // 自适应 block size

fake_quant_fwd_vec2_persistent<<<grid, block, 0, stream>>>(...);
```

#### 范式3：任务获取辅助函数

```cpp
// 4. 原子操作获取下一个任务
__device__ __forceinline__ int persistent_next(int* counter) {
    // 1 CTA fetches 1 task (group)
    return atomicAdd(counter, 1);
}

// 可选：一次获取多个任务以减少原子操作次数
template<int CHUNK>
__device__ __forceinline__ int persistent_next_chunk(int* counter) {
    return atomicAdd(counter, CHUNK);
}
```

#### 范式4：Kernel 主循环结构

```cpp
// 5. Persistent Kernel 核心结构
template<typename...>
__global__ void fake_quant_fwd_vec2_persistent(
    const XType* __restrict__ x,
    YType* __restrict__ y,
    const ScaleType* __restrict__ scale,
    const ZpType* __restrict__ zp,
    int32_t qmin, int32_t qmax,
    int G,              // Group size
    int num_tasks,      // 总任务数（N_groups）
    int* __restrict__ counter  // 任务计数器
) {
    int tid = threadIdx.x;

    // 共享变量：存储当前任务的组索引和参数
    __shared__ int g_sh;       // 当前处理的 group 索引
    __shared__ float s_sh;     // 共享的 scale
    __shared__ float inv_s_sh; // 共享的 1/scale
    __shared__ float zpf_sh;   // 共享的 zero_point

    // ========== 主循环：持续获取并处理任务 ==========
    while (true) {
        // 步骤1：只有 tid==0 的线程负责获取任务
        if (tid == 0) {
            g_sh = persistent_next(counter);
        }
        __syncthreads();  // 同步所有线程，确保 g_sh 对所有线程可见

        // 步骤2：检查是否所有任务已完成
        int g = g_sh;
        if (g >= num_tasks) break;  // 退出循环

        // 步骤3：由 tid==0 的线程加载当前任务的参数到共享内存
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
        __syncthreads();  // 同步，确保参数加载完成

        // 步骤4：所有线程从共享内存读取参数，处理当前任务
        float s = s_sh;
        float inv_s = inv_s_sh;
        float zpf = zpf_sh;

        // 处理当前 group 的数据
        int base = g * G;
        int nvec = G / 2;

        #pragma unroll
        for (int vi = tid; vi < nvec; vi += blockDim.x) {
            int idx = base + vi * 2;

            // 加载数据、计算、存储
            VX xv = Vec2<XType>::load2(reinterpret_cast<const XType*>(x + idx));
            // ... 计算逻辑 ...
            Vec2<YType>::store2(reinterpret_cast<YType*>(y + idx), yv);
        }

        // 步骤5：同步，确保所有线程完成当前任务后再进入下一轮
        __syncthreads();
    }
    // ========== 主循环结束 ==========
}
```

#### 范式5：关键设计要点

| 要点 | 说明 | 代码示例 |
|------|------|----------|
| **原子任务获取** | 使用 `atomicAdd` 原子操作，避免竞争 | `atomicAdd(counter, 1)` |
| **单线程获取** | 只有 `tid==0` 获取任务，减少原子操作 | `if (tid == 0) g_sh = persistent_next(counter);` |
| **共享内存同步** | 每次任务获取后必须 `__syncthreads()` | `__syncthreads();` |
| **退出条件** | 当任务ID超过总数时退出 | `if (g >= num_tasks) break;` |
| **参数复用** | group 参数加载到共享内存，所有线程复用 | `__shared__ float s_sh, inv_s_sh, zpf_sh;` |

#### 范式6：适用场景

Persistent Kernel 适用于以下场景：

1. **任务数远大于 SM 数量**（如 `N_groups = 1,000,000`）
2. **每个任务计算量相对较小**（单个 group 量化）
3. **任务之间相互独立**（无数据依赖）
4. **需要动态负载均衡**（避免尾效应）

#### 范式7：反向 kernel 的 persistent 模式

反向 kernel 同样采用 persistent 模式，但在任务处理完成后需要额外的归约步骤：

```cpp
__global__ void fake_quant_bwd_vec2_persistent(...) {
    // ... 前面的任务获取和参数加载逻辑相同 ...

    while (true) {
        if (tid == 0) g_sh = persistent_next(counter);
        tb.sync();
        int g = g_sh;
        if (g >= num_tasks) break;

        // ... 参数加载逻辑 ...

        // 累加局部梯度
        float local_dscale = 0.0f;
        float local_dzp = 0.0f;

        #pragma unroll
        for (int vi = tid; vi < nvec; vi += blockDim.x) {
            // ... 计算局部梯度 ...
            local_dscale += ...;
            if constexpr (HasZp) local_dzp += ...;
        }

        // 归约：将局部梯度汇总到全局内存
        float sum_dscale = block_reduce_sum_cg_lane0(local_dscale);
        float sum_dzp = 0.0f;
        if constexpr (HasZp) sum_dzp = block_reduce_sum_cg_lane0(local_dzp);

        // 只有一个线程负责写回
        if (warp == 0 && lane == 0) {
            dscale[g] = from_f32<ScaleType>(sum_dscale);
            if constexpr (HasZp) dzp[g] = from_f32<ZpType>(sum_dzp);
        }

        tb.sync();  // 同步后进入下一轮任务
    }
}
```

### 机制（你这份代码真实行为）

每个 CTA（block）在 `while(true)` 里反复做：

1. `tid==0` 领取一个 group：`g_sh = atomicAdd(counter,1)`
2. 全 block 同步，读取 `g_sh`
3. 若 `g >= N_groups` 则退出
4. 处理该 group 的 Vec2 循环
5. `__syncthreads()`，进入下一轮

### 具体案例 1：N_groups 巨大，避免"百万 blocks launch"

假设：

* `G=128`
* `x.numel = 128 * 1,000,000` → `N_groups=1,000,000`

如果用传统 grid = `N_groups`：

* 你要 launch **一百万个 blocks**（即使每个 block 很轻，也会在 launch/scheduling 上出很多额外开销和调度压力）

而你的 persistent 做法：

* 只 launch `num_sms * 2` 个 blocks
  比如某张卡 `num_sms=108`（示例值），则只启动 **216 blocks**
* 这 216 个 blocks 在 while 循环里把 1,000,000 个 group 领完

**这里的"优化收益"非常具体**：不是"尾效应"抽象概念，而是**把 grid 从 1,000,000 缩到 ~200**，launch/scheduling 负担从量级上被砍掉。

### 具体案例 2：N_groups 中等但不整齐，减少尾巴空转

假设：

* `num_sms=80`（示例）
* `resident_ctas=160`
* `N_groups=10,003`

传统 grid=10,003 时，尾部 wave 往往会出现 "最后一波只剩少量 blocks 在跑，其余 SM 空转"。

你的 persistent 版本里：

* 每个 block 做完一个 group 立即领下一个
* 最终会收敛到少量 blocks 跑最后几个 group，但这时其它 blocks 也会更快退出（不会像固定 grid 那样出现长尾拖着一堆 SM idle）

> 你这份代码的设计目标更偏"**减少 launch 开销 + 动态均衡**"，尤其在 group 数大时收益最显著。

---

## 2. 优化点 #2：Vec2 向量化访存——"每线程一次处理 2 个元素"，并保证对齐

### 对应代码锚点

* Vec2 定义：`Vec2<half>`, `Vec2<__nv_bfloat16>`, `Vec2<float>`
* 前向/反向都用：

  ```cpp
  int nvec = G / 2;
  for (int vi = tid; vi < nvec; vi += blockDim.x) {
      int idx = base + vi * 2;
      VX xv = Vec2<XType>::load2(x + idx);
      ...
      Vec2<YType>::store2(y + idx, yv);
  }
  ```

### 机制（你这份代码为什么"真的能 vectorize"）

你这里 `idx = base + vi*2`，而 `base = g*G`，并且你强制 `G ∈ {64,128,256}`：

* `G` 永远是偶数 → `base` 永远是偶数 → `idx` 永远是偶数
* 对 `half2/bfloat162`：2 字节元素 * 偶数 index → 4 字节对齐通常成立
* 对 `float2`：4 字节元素 * 偶数 index → 8 字节对齐成立

也就是说，你这份实现不仅写了 `reinterpret_cast<half2*>`，而且**用 index 设计确保了对齐条件**，这点很关键（否则 vector load 可能退化或触发未对齐访问问题）。

### 具体案例：G=128 时，访存完全整齐

* `G=128 → nvec=64`
* `blockDim = pick_block(128)`（下面会算）通常是 64 threads
* 于是每个线程 `tid` 只跑一次循环：`vi=tid`
* warp 内访问：

  * thread0 读 `x[base+0..1]`
  * thread1 读 `x[base+2..3]`
  * ...
  * thread31 读 `x[base+62..63]`

这是**完美的连续 coalesced + vectorized**：每个线程一次 2 元素，warp 覆盖连续区间。

---

## 3. 优化点 #3：共享内存缓存 group 参数（scale / inv_scale / zp）——避免"每元素重复读参数 + 重复除法"

### 对应代码锚点

前向 persistent 的 group 参数准备：

```cpp
__shared__ float s_sh, inv_s_sh, zpf_sh;
if (tid == 0) {
    float raw_s = to_f32(scale[g]);
    float s_local = clamp_abs_sign(raw_s, 1e-5f, 1e4f);
    s_sh = s_local;
    inv_s_sh = 1.0f / s_local;
    if constexpr (HasZp) { ... zpf_sh = rounded_zp; }
}
__syncthreads();
```

反向同理（并且保持与 forward 同样的 clamp/round 语义）。

### 具体案例：G=256 时"参数复用次数"非常直观

* `G=256`，一个 group 有 256 个元素
* 如果不缓存：

  * 每个元素都需要读 `scale[g]`（以及 `zp[g]`），等价于 **每 group 256 次读 scale**
  * 并且每个元素都要做 `x / s`（除法）
* 你现在的做法：

  * 每 group **只读 1 次 scale（和 1 次 zp）**
  * 每 group **只算 1 次 `inv_s = 1/s`**
  * 组内 256 个元素全部用乘法 `x * inv_s`

这不是"抽象的减少访存"，而是**把 per-group 参数读取次数从 O(G) 变成 O(1)**，并把 G 次除法变成 G 次乘法。

---

## 4. 优化点 #4：block size 自适应（pick_block）——让"每线程负载接近 1 次循环"，减少空转与分支

### 对应代码锚点

```cpp
static inline int pick_block(int G) {
    int nvec = G / 2;
    int b = 32;
    while ((b << 1) <= nvec) b <<= 1;
    return b;
}
```

### 具体案例：三种 group_size 对应的 blockDim 是确定的

* `G=64  → nvec=32  → block=32`
* `G=128 → nvec=64  → block=64`
* `G=256 → nvec=128 → block=128`

于是你的 Vec2 循环：

```cpp
for (vi = tid; vi < nvec; vi += blockDim.x)
```

在这三种配置下几乎都是 **每线程只执行 1 次**（tid 刚好覆盖 0..nvec-1），这会带来两个很实际的好处：

1. loop 开销/分支判断极少
2. warp 内线程工作量非常均匀，不容易出现"部分线程空跑多次循环"的不平衡

---

## 5. 优化点 #5：反向归约（dscale/dzp）——用 warp+shared 做"每 group 只写一次全局内存"

### 对应代码锚点

* 每线程累加局部量：

  ```cpp
  float local_dscale = 0.0f;
  float local_dzp = 0.0f;
  // loop 内不断 +=
  ```
* block 归约函数：

  ```cpp
  float sum_dscale = block_reduce_sum_cg_lane0(local_dscale);
  if constexpr(HasZp) sum_dzp = block_reduce_sum_cg_lane0(local_dzp);
  ```
* 最终写回：

  ```cpp
  if (warp==0 && lane==0) dscale[g] = ...; dzp[g] = ...;
  ```

### 具体案例：G=128 时，归约层级与写回次数一眼可数

* `G=128 → blockDim=64 → 2 个 warp`
* `local_dscale` 每个线程只算 2 个元素（Vec2），然后参与归约
* `warp_sums[8]` 的 shared buffer 实际只用到前 2 个槽位（warp0, warp1）
* 最后 **整个 group 的 dscale 只发生 1 次全局写**（同样 dzp 也 1 次）

这比"每个元素 atomicAdd 到全局 dscale[g]"的朴素写法，写冲突/序列化会少一个数量级。

> 你这里 `warp_sums[8]` 的设计也和 `pick_block()` 形成闭环：最大 blockDim=128 → 4 warps，远小于 8，安全且不浪费太多 shared。

---

## 6. 优化点 #6：STE mask 与 PyTorch 行为对齐——"mask 取决于 round 后、clamp 前的值"

### 对应代码锚点

```cpp
float u0 = x0 * inv_s + zpf;
int32_t q0_unclamped = (int32_t)round_bankers(u0);
float m0 = (q0_unclamped >= qmin && q0_unclamped <= qmax) ? 1.0f : 0.0f;
int32_t q0 = clamp_i32(q0_unclamped, qmin, qmax);

float dx0 = g0 * m0;
local_dscale += g0 * ( ((float)q0 - zpf) - m0 * (x0 * inv_s) );
if constexpr(HasZp) local_dzp += g0 * s * (m0 - 1.0f);
```

### 具体案例：边界外样本对 dx 的影响（非常具体）

设：

* `qmin=-128, qmax=127`（举例）
* 某个元素算出来 `u0=200.3`
* `round_bankers(u0)=200` → `q0_unclamped=200`
* 因为 200 > 127 → `m0=0`
* clamp 后 `q0=127`

于是：

* `dx0 = dy * 0 = 0`（梯度被截断，不回传到 x）
* 但 `dscale/dzp` 仍然会按公式累计（因为 scale/zp 要对饱和行为负责）

这正是你注释里说的 "Match PyTorch STE: mask based on rounded pre-clamp"。

---

## 7. 实现层面的"真实结论"（结合你代码能得出的）

1. **你现在的 wrapper 永远走 persistent**
   forward 调用的是 `fake_quant_fwd_vec2_persistent`，backward 调用的是 `fake_quant_bwd_vec2_persistent`。
   非 persistent 版本存在，但当前路径不使用（更多像对照/调试保留）。

2. **你的三档 group_size（64/128/256）与 pick_block 完全匹配**
   这使得 Vec2 循环基本"一线程一迭代"，结构非常干净。

3. **Vec2 的对齐在你的约束下成立**
   因为 `idx` 强制为偶数，且 `base=g*G`、`G` 为 64/128/256，避免了最常见的未对齐 vector load 风险。

4. **归约路径的 warp_sums[8] 与 blockDim 上界一致**
   你没有让 blockDim 超过 128（最多 4 warps），不会踩 shared 越界。

