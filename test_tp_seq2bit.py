#!/usr/bin/env python3
"""TP=2 Test for Seq2Bit QAT - Run with: torchrun --nproc_per_node=2 test_tp_seq2bit.py"""

import sys
import os
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410")
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410/Megatron-LM")

import torch
import torch.distributed as dist

def main():
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] Starting TP test with world_size={world_size}")

    # Initialize Megatron parallel state
    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
    )

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_world = parallel_state.get_tensor_model_parallel_world_size()
    print(f"[Rank {rank}] TP rank={tp_rank}, TP world={tp_world}")

    # Test Seq2Bit quantizer
    from megatron.core.quantization.megatron_qat import MegatronSeq2BitQuantizer, QuantConfig

    # Create a simple weight
    hidden_size = 256
    weight = torch.randn(hidden_size, hidden_size, device='cuda')

    config = QuantConfig(quant_type="seq2bit", group_size=64)
    quantizer = MegatronSeq2BitQuantizer(weight, config, prefix="test")
    quantizer.use_weight_quant = True

    # Quantize
    weight_q = quantizer(weight)

    print(f"[Rank {rank}] Weight quantized: shape={tuple(weight_q.shape)}, finite={torch.isfinite(weight_q).all().item()}")

    # Cleanup
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

    print(f"[Rank {rank}] TP test complete")

if __name__ == "__main__":
    main()
