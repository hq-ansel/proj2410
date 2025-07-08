
cccl (CUDA c++ core Libraries) 重构来自三个项目
Thrust,CUB,libcudacxx
整个cccl是为了实现高质量，高性能，且易用的CUDA编程接口。

libcu++ 是 CUDA C++ 标准库。它提供了一种既能在主机代码中运行又能在设备代码中运行的 C++ 标准库实现。此外，它还为 CUDA 特有的硬件特性（如同步原语、缓存控制、原子操作等）提供了抽象接口。

CUB 是一个低级别的、专为 CUDA 设计的库，旨在为所有 GPU 架构提供闪电般速度的并行算法。除了设备级算法外，它还提供了协作式算法，如块级归约和线程束级扫描，为 CUDA 内核开发者提供了构建块，以便创建闪电般速度的自定义内核。

Thrust 是 C++ 并行算法库，它启发了 C++ 标准库中并行算法的引入。Thrust 的高级接口极大地提高了程序员的生产力，同时通过可配置的后端实现了在 GPU 和多核 CPU 之间的性能可移植性，这些后端允许使用多种并行编程框架（如 CUDA、TBB 和 OpenMP）。

Cuda Experimental 是一个实验性功能库，这些功能仍处于设计阶段。

CCCL C++ 库的主要目标是填补标准 C++ 库为标准 C++ 所扮演的角色：为 CUDA C++ 开发者提供通用的、闪电般速度的工具，使他们能够专注于解决重要问题。将这些项目统一起来是实现该目标的第一步。

## libcu++/libcudacxx

libcudaxx 提供最基本的c++抽象，

提供：
- 可以在host和device上运行的c++标准库
- 对c++标准库的扩展
- 基本的CUDA编程模型抽象

### c++标准库扩展

默认情况下c++标准库是不支持CUDA的，因为比如`std::string`,`std::vector`这样的容器没有`__host__`或`__device__`修饰符，所以无法在device上运行。

libcudacxx的目标是为了提供一个异构的c++
- 按需使用(opt-in) 不会替换由host编译器提供的标准库
- 渐近实现(Incremental) 不提供完整的c++标准库实现，而是针对CUDA环境进行优化和扩展
- 异构支持(Heterogeneous) 既可以在主机端(host)上运行，也可以在设备端(device)上运行，且支持在主机和设备之间传递数据

使用libcudacxx非常简单,将原来的`<atomic>`替换为`<cuda/std/atomic>`即可。
并且使用时将命名空间`std`改为`cuda::std`
```c++
#include <cuda/std/atomic>
cuda::std::atomic<int> x;
```

并且libcudacxx没有独立文档，libcu++ 不重复编写标准库（如 std::vector、std::atomic）的常规说明，避免冗余。

例如，若需了解 std::vector 的用法，仍需参考标准 C++ 文档（如 cppreference）。

libcu++ 仅明确列出 ​​哪些标准库头文件被适配并可在 CUDA 中使用​​（如 <vector>、<atomic>）。

对于头文件内的具体功能（如 std::vector::push_back），开发者需自行查阅标准文档。

### c++ 标准库扩展

对于某些纯c++标准库无法实现的高性能CUDA C++代码，libcudacxx 提供了相应的扩展。

例如libcudacxx扩展了`cuda::atomic<T>`，和其他的同步原语，使其可以在线程的范围内空置内存屏障，为了使用扩展的功能需要去除命名空间`std`,例如
```c++
#include <cuda/atomic>
cuda::atomic<int, cuda::thread_scope_device> x;
```

### CUDA编程模型抽象

某些编程抽象和c++标准库并不完全等价，例如`cuda::memcpy_async`是一个关键的异步数据移动抽象，主要在设备的全局内存和共享内存之间进行数据传输。这个抽象的具体硬件指令实际是LDGSTS指令，主要在Ampere和Hopper的Tensor Memory Accelerator上有实现。

### 总结`std::`,`cuda::`,和`cuda::std::`命名空间的区别

- `std::`/`<*>` 是主机端的编译器标准库，仅仅在`__host__`代码中使用, 但是也可以通过`--expt-relaxed-constexpr`编译选项在device使用任意被`constexpr`的函数,libcudacxx不会替换host编译器的任何c++标准库实现。
- `cuda::std::`/`<cuda/std/*>` 是CUDA C++标准库的扩展，在主机端和设备端都可以使用
- `cuda::`/`<cuda/*>` c++标准库的扩展，主机端和设备端都可以使用
- `cuda::device`/`cuda/device/*` 设备端的c++标准库实现，仅在device代码中使用
- `cuda::ptx` 用于inline PTX的C++包装器，仅在device代码中使用

```c++
// Standard C++, __host__ only.
#include <atomic>
std::atomic<int> x;

// CUDA C++, __host__ __device__.
// Strictly conforming to the C++ Standard.
#include <cuda/std/atomic>
cuda::std::atomic<int> x;

// CUDA C++, __host__ __device__.
// Conforming extensions to the C++ Standard.
#include <cuda/atomic>
cuda::atomic<int, cuda::thread_scope_block> x;
```

### libcudacxx的要求

建议c++ 17以上，

### 