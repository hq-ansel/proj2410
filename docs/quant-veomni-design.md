# VeOmni 量化训练与并行架构（train.py）

面向“可直接转化为架构图/流程图”的文字化描述，突出 VeOmni 在量化训练中的多并行特性（DP/TP/PP/EP/CP/Ulysses/FSDP）与量化器接入链路。

```mermaid
flowchart TD
    subgraph Init[并行初始化]
        A[CLI/YAML -> parse_args\nArguments+QuantizerArguments] --> B[dist.init_process_group]
        B --> C[init_parallel_state\nDP+DP_replicate+DP_shard+TP+PP+EP+CP+Ulysses]
    end

    subgraph Data[数据路径]
        C --> D1[build_tokenizer / build_chat_template]
        D1 --> D2[build_*_dataset\niterable/mapping/interleave/energon]
        D2 --> D3[build_dataloader\nbsz warmup,rmpad,prefetch]
    end

    subgraph ModelQuant[模型与量化接入]
        C --> M1[build_foundation_model\nload config/weights (float)]
        M1 --> M2[EQuantConfig <- QuantizerArguments\nn_bits/group_size/clamp/round/...]
        M2 --> M3[convert_linear\nnn.Linear -> IntQuantLinear/QuantLinearFake]
        M3 --> M4[build_parallelize_model\nFSDP shard + TP slice + PP stage\n可选 gradient ckpt / activation offload]
    end

    subgraph QuantCtrl[量化控制]
        Q1[use_weight_quant=False 预热] --> Q2[暖启动后 set_quant_state(weight=True)]
        Q2 --> Q3[GradualQuantContext.step\nquantization_position_ratio]
        Q3 --> Q4[iterative_freezing/is_tracking\nfreeze_momentum/threshold]
        Q4 --> Q5[optimizer param groups\n主权重 vs scale/zero_point(lr 独立, wd=0)]
    end
    M3 --> Q1
    Q5 --> T5

    subgraph Train[训练循环（每 global_step）]
        T1[迭代 dataloader -> micro_batches\nUlysses 切 micro-batch]
        T2[model_fwd_context forward\n在 PP stage + TP shard 内运行]
        T3[model_bwd_context backward\nFSDP 梯度重计算/激活重计算]
        T4[clip_grad_norm_] --> T5[optimizer.step + lr_scheduler.step + zero_grad]
        T6[all_reduce(loss, grad_norm) on fsdp_group]
        T7[environ_meter / wandb log]
        T8[Checkpointer 分布式保存\nsave_steps/save_epochs]
    end
    D3 --> T1
    M4 --> T2
    Q2 --> T2
    T5 --> T6 --> T7 --> T8
```

## 结构化文字流程（可直接绘图）
1) 初始化与并行拓扑  
   - `init_process_group` 后调用 `init_parallel_state`，一次性确定 DP/DP_replicate/DP_shard、TP、PP、EP、CP、Ulysses 的尺寸与通信组。  
   - 所有后续模块（dataloader、模型、优化器、checkpoint）共享该并行状态，通过 `get_parallel_state().fsdp_group` 等接口执行 all-reduce、shard 和同步。

2) 数据侧（与并行解耦）  
   - tokenizer/chat_template 构建只做在 rank0/全局共享。  
   - dataset 选择 iterable/mapping/interleave/energon，dataloader 内含 bsz warmup、rmpad、prefetch；输出的 micro-batches 会再按 Ulysses pipeline 微切，供 PP+TP stage 并行消费。

3) 模型与量化接入链路  
   - `build_foundation_model` 先加载 float 权重/配置；保持 init_device 流程（meta/cuda）以兼容并行包装。  
   - 将 `QuantizerArguments` 映射到 `EQuantConfig`（量化类型、比特数、组大小、clamp/round、渐进/冻结参数、LoRA rank 等）。  
   - `convert_linear` 全量遍历线性层，替换为量化版本并挂载 `weight_quantizer`，保留原 dtype/设备；默认可跳过 Embedding/头部，后续可通过名单或正则精细选择。  
   - `build_parallelize_model` 在量化后的模型上叠加并行：  
     - FSDP shard：按 DP shard 划分参数，支持 offload；  
     - TP：线性层维度切分，与量化权重 shard 对齐；  
     - PP：拆分为多个 pipeline stage，micro-batch 按 stage 流动；  
     - CP/EP：上下文/专家并行保持量化权重一致性；  
     - 可选 gradient checkpoint 与 activation offloading（由 `build_activation_offloading_context` 提供 fwd/bwd ctx）。

4) 量化控制策略  
   - 预热：`use_weight_quant=False`，先稳定浮点训练。  
   - 切换：到 warmup 步后 `set_quant_state(weight=True)` 开启假量化；若量化类型为 gradual，则使用 `GradualQuantContext.step(global_step)` 推进 `quantization_position_ratio`。  
   - 稳定：`iterative_freezing` + `is_tracking` 结合 `freeze_momentum/threshold` 逐层冻结稳定分组，减少抖动。  
   - 优化器参数组：建议将量化参数（scale/zero_point/LoRA rank）独立 lr、`weight_decay=0`；可选择仅训量化参数或与主权重共训。

5) 训练循环与并行执行  
   - dataloader 产出 micro-batches -> Ulysses 切分 -> 送入当前 PP stage + TP shard；forward/backward 包裹在 offload/ckpt ctx 内。  
   - 梯度裁剪优先使用模型自带 `clip_grad_norm_`（已注册 FSDP/TP 感知版）。  
   - `optimizer.step` + `lr_scheduler.step` 后，`all_reduce` 在 `fsdp_group` 聚合 loss/grad_norm，保证 DP shard 之间一致。  
   - `environ_meter` 记录吞吐/latency，rank0 用 wandb 记录；按 `save_steps/save_epochs` 用 Checkpointer 分布式保存，可恢复 lr_scheduler/dataloader/environ_meter 状态。

6) 保存与导出  
   - 结束后可将最后一次分布式 checkpoint 转为 HF 权重（`save_model_weights`），保持量化权重/配置与 tokenizer/chat_template 打包。

## 关键并行特性速览
- 统一并行入口：`init_parallel_state` 一次性创建 DP/TP/PP/EP/CP/Ulysses/FSDP 拓扑，后续组件均读取同一并行上下文。  
- 量化后再并行：先 `convert_linear` 再 `build_parallelize_model`，确保 shard/切分时量化权重已到位，避免多次替换。  
- 混合并行协同：TP 切分线性权重、FSDP shard 参数、PP 分 stage；all-reduce/gradient-clip 均使用并行感知实现。  
- 内存优化：gradient checkpoint、activation offload、FSDP offload 可与量化同时启用，减轻大模型 QAT 的显存压力。  
- 恢复一致性：checkpoint 包含并行配置与 dataloader/environ_meter 状态，resume 后量化/并行开关保持一致。
