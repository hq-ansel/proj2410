# Pipeline Parallelism (PP) Implementation Plan

## 1. Overview
This document outlines the roadmap for integrating Pipeline Parallelism (PP) into the `VeOmni/veomni/distributed` module. The goal is to enable multi-node, large-scale model training by splitting model layers across stages (ranks) while maintaining compatibility with existing FSDP/TP/SP strategies.

## 2. Existing Components (Reuse)
We will leverage the following existing components to maintain consistency with the current architecture:

### `parallel_state.py`
This file is PP-ready and will serve as the source of truth for topology.
*   **Reuse `ParallelState` attributes:**
    *   `pp_size`, `pp_rank`, `pp_group`, `pp_mesh`.
    *   `is_first_pp_stage`, `is_last_pp_stage`: Critical for handling input/output conditional logic (e.g., loss calculation only on the last stage).
*   **Reuse `init_parallel_state`:**
    *   Already handles `DeviceMesh` creation for PP (`mesh_dim_names` includes "pp").
    *   Ensures valid global rank mapping.

### `torch_parallelize.py`
*   **Integration Point:** The entry point `build_parallelize_model` currently handles FSDP and TP. PP wrapping logic needs to be inserted here, likely **before** FSDP wrapping (as FSDP usually wraps layers *within* a PP stage).

## 3. New Components Required
We recommend creating a new submodule `VeOmni/veomni/distributed/pipeline/` to house PP-specific logic.

### 3.1 Directory Structure Proposal
```text
VeOmni/veomni/distributed/
├── pipeline/                <-- NEW
│   ├── __init__.py
│   ├── partition.py         # Logic to split models (e.g., Transformers) into stages
│   ├── runtime.py           # Schedule execution (1F1B, Interleaved)
│   ├── p2p.py               # Point-to-Point communication primitives (Send/Recv)
│   └── stage.py             # PipelineStage wrapper module
```

### 3.2 Component Details

#### A. `partition.py` (Model Partitioning)
*   **Responsibility:** Given a global model (e.g., Llama, Qwen) and `pp_size`, slice the `nn.ModuleList` (decoder layers) into chunks.
*   **Strategy:**
    *   First stage: Embeddings + Layers [0:N]
    *   Middle stages: Layers [N:M]
    *   Last stage: Layers [M:End] + RMSNorm + LM_Head
*   **Challenge:** Must handle "tied weights" (Embedding sharing with LM_Head) appropriately, typically by keeping copies or communicating gradients.

#### B. `stage.py` (Stage Wrapper)
*   **Responsibility:** Wrap the local partition of the model into a `PipelineStage` module.
*   **Functionality:**
    *   Define input/output shapes for tensors passing between stages.
    *   Handle moving inputs to device.

#### C. `p2p.py` (Communication)
*   **Responsibility:** Wrap `torch.distributed.batch_isend_irecv` or `dist.send/recv`.
*   **Key Functions:**
    *   `send_forward()`, `recv_forward()`
    *   `send_backward()`, `recv_backward()`
    *   Must respect `parallel_state.pp_group`.

#### D. `runtime.py` (Schedules)
*   **Responsibility:** Orchestrate the micro-batch execution.
*   **Implementation Options:**
    1.  **Native Implementation:** Implement `1F1B` (One-Forward-One-Backward) schedule manually using `p2p.py`.
    2.  **Torch Pipelining (Recommended):** Wrap `torch.distributed.pipelining` (available in PyTorch > 2.4) if the environment allows. This abstracts standard schedules.

## 4. Integration Logic (Draft)

In `torch_parallelize.py`:

```python
def parallelize_model_pp(model, ...):
    parallel_state = get_parallel_state()
    
    # 1. Partition the model
    # This might require modifying the model structure in-place or returning a submodule
    local_stage_model = partition_model(model, parallel_state.pp_rank, parallel_state.pp_size)
    
    # 2. Apply FSDP/TP to the *local* stage modules
    # FSDP should only shard the parameters residing on this rank
    local_stage_model = parallelize_model_fsdp1(local_stage_model, ...) 
    
    # 3. Create Pipeline Schedule Runtime
    # This object will replace the standard forward/backward loop in train.py
    pipeline_runtime = PipelineRuntime(local_stage_model, ...)
    
    return pipeline_runtime
```

## 5. Next Steps
1.  **Scaffold `distributed/pipeline/` directory.**
2.  **Implement `p2p.py`:** Verify basic tensor transmission across ranks using `pp_group`.
3.  **Implement `partition.py`:** Create a splitter specifically for the Transformer architecture used in VeOmni (likely Llama/Qwen based).
4.  **Implement `1F1B` Schedule:** Connect P2P ops with forward/backward passes.

---

# TP + PP Compatibility Plan (Prioritize QAT)

## 6. Scope & Priority
Primary target: **`tasks/quantize/train.py`** (QAT + KD).  
Secondary target: `tasks/train_torch.py` (standard LM).

**Assumptions / constraints (initial):**
*   **TP size > 1** is allowed only when **PP size > 1** is also enabled.
*   **Data parallel mode must be DDP** (no FSDP1/2 while TP+PP is being validated).
*   **Fixed shapes only** (no `rmpad`, no dynamic batch size) to keep P2P tensor shapes deterministic.
*   **LLM-style causal LM only** (Qwen/Llama‑like), no diffusion or multimodal variants in phase 1.

## 7. Key Design Decisions
1.  **Order of parallelization**
    *   Partition by PP first → apply TP within each local stage.
    *   This keeps PP send/recv boundaries clean and localizes TP to stage parameters.

2.  **PP tensor shapes under TP**
    *   `pp_input_shape` must describe the **local TP shard** shape.
    *   Add a helper that computes TP‑local hidden shape based on `hidden_size` and TP policy (e.g., column‑parallel vs row‑parallel).

3.  **Loss computation at last PP stage**
    *   With TP‑sharded logits, loss must be **TP‑aware**.
    *   Option A (simple): all‑gather logits across TP ranks, then compute loss.
    *   Option B (preferred): use TP‑aware fused loss if available.

4.  **P2P communication**
    *   P2P must use the **PP group only**; TP all‑gathers must use TP group.
    *   Ensure PP send/recv is done on **local TP shard** tensors to avoid extra all‑gathers.

## 8. Implementation Plan (Quantize‑first)

### 8.1 Enable TP+PP in config/args
*   Remove the hard assert that blocks `tensor_parallel_size > 1`.
*   Add runtime checks:
    *   `tp_size > 1` requires `pp_size > 1`.
    *   Force `data_parallel_mode == "ddp"`.
    *   Disallow `rmpad` and `dyn_bsz`.

### 8.2 TP‑aware PP input shape
*   Add `infer_pp_input_shape_tp(model, micro_batch_size, max_seq_len, tp_size, tp_policy)`:
    *   Determine whether hidden size is sharded (row/col parallel).
    *   Return **TP‑local** tensor shape used by `recv_forward`.

### 8.3 Pipeline runtime extensions
*   Add TP‑aware loss path:
    *   If TP enabled, gather logits (or use fused loss) before CE.
*   Add optional hooks to allow QAT‑specific loss composition:
    *   Accept a `loss_fn` callback in `forward_backward_1f1b`.
    *   This lets QAT/KD compute custom loss while PP controls scheduling.

### 8.4 Quantize training integration
*   In `tasks/quantize/train.py`:
    *   Pass TP‑aware `pp_input_shape`.
    *   Replace the per‑micro‑batch backward loop with:
        *   `forward_backward_1f1b(micro_batches, loss_fn=quant_kd_loss_fn)`
    *   Ensure teacher model uses **same PP/TP topology** or is disabled for the first phase.

### 8.5 Validation steps
*   **Unit smoke test**: 2×PP × 2×TP on a tiny Qwen model with fixed sequence length.
*   Verify:
    *   All ranks progress (no deadlock).
    *   PP send/recv shapes match TP shard sizes.
    *   Loss numerics are stable vs TP‑only baseline.

## 9. Incremental Rollout
1.  **Phase 1**: QAT (no KD), TP+PP, DDP only, fixed shapes.
2.  **Phase 2**: Enable KD teacher (PP+TP aware) or keep teacher on PP‑only path.
3.  **Phase 3**: Extend to standard `train_torch.py` and other LM tasks.
4.  **Phase 4**: Evaluate compatibility with FSDP2 and activation offload.
