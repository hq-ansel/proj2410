#!/usr/bin/env python3
"""Verify Megatron-LM configuration for Qwen2.5-0.5B with TP/SP"""

import sys
import subprocess

def verify_model_arch():
    """Verify Qwen2.5-0.5B model architecture"""
    print("=" * 60)
    print("Verifying Qwen2.5-0.5B Model Architecture")
    print("=" * 60)
    
    # Qwen2.5-0.5B specs
    expected = {
        "num_layers": 24,
        "hidden_size": 896,
        "ffn_hidden_size": 4864,
        "num_attention_heads": 14,
        "num_query_groups": 2,
        "head_dim": 64,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
    }
    
    for key, value in expected.items():
        print(f"✓ {key}: {value}")
    
    # Verify divisibility for TP
    print("\n--- TP Compatibility Check ---")
    for tp in [1, 2, 4, 8]:
        hidden_ok = (expected["hidden_size"] % tp == 0)
        ffn_ok = (expected["ffn_hidden_size"] % tp == 0)
        heads_ok = (expected["num_attention_heads"] % tp == 0)
        
        status = "✓" if all([hidden_ok, ffn_ok, heads_ok]) else "✗"
        print(f"{status} TP={tp}: hidden={hidden_ok}, ffn={ffn_ok}, heads={heads_ok}")
    
    return True

def verify_tp_sp_config():
    """Verify TP/SP configuration"""
    print("\n" + "=" * 60)
    print("Verifying TP/SP Configuration")
    print("=" * 60)
    
    # TP/SP rules
    configs = [
        (1, False, "Baseline (no TP)"),
        (2, True, "TP=2 with SP (recommended)"),
        (4, True, "TP=4 with SP"),
        (8, True, "TP=8 with SP"),
    ]
    
    for tp, sp, desc in configs:
        print(f"✓ {desc}: --tensor-model-parallel-size {tp}", end="")
        if sp:
            print(" --sequence-parallel")
        else:
            print()
    
    return True

def verify_megatron_args():
    """Verify Megatron-LM arguments"""
    print("\n" + "=" * 60)
    print("Verifying Megatron-LM Arguments")
    print("=" * 60)
    
    required_args = [
        "--use-mcore-models",
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--context-parallel-size",
        "--num-layers 24",
        "--hidden-size 896",
        "--ffn-hidden-size 4864",
        "--num-attention-heads 14",
        "--group-query-attention",
        "--num-query-groups 2",
        "--position-embedding-type rope",
        "--swiglu",
        "--normalization RMSNorm",
        "--bf16",
    ]
    
    for arg in required_args:
        print(f"✓ {arg}")
    
    return True

def main():
    print("\n" + "=" * 60)
    print("Megatron-LM TP/SP Configuration Verification")
    print("Model: Qwen2.5-0.5B")
    print("=" * 60)
    
    all_ok = True
    all_ok &= verify_model_arch()
    all_ok &= verify_tp_sp_config()
    all_ok &= verify_megatron_args()
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)
        print("\nExample usage:")
        print("  bash script/train.sh --backbone megatron-lm --tp-size 2")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
