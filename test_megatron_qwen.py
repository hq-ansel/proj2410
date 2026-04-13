#!/usr/bin/env python3
"""
Test script: Verify Megatron-LM can work with Qwen2.5-0.5B
"""
import os
import sys
import torch

# Add Megatron-LM to path
sys.path.insert(0, '/home/ubuntu/data/exp/proj2410/Megatron-LM')

from transformers import AutoTokenizer, AutoConfig
from megatron.core import parallel_state

def test_basic_imports():
    """Test basic imports work"""
    print("=" * 50)
    print("Step 1: Testing basic imports")
    print("=" * 50)

    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.models.gpt.gpt_model import GPTModel
        print("✓ Megatron core modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Megatron modules: {e}")
        return False

    return True

def test_tokenizer():
    """Test Qwen2.5-0.5B tokenizer can be loaded"""
    print("\n" + "=" * 50)
    print("Step 2: Testing tokenizer loading")
    print("=" * 50)

    model_path = "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
        print(f"  Vocab size: {len(tokenizer)}")

        # Test encoding
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        print(f"  Test encode: '{text}' -> {tokens}")

        # Test decoding
        decoded = tokenizer.decode(tokens)
        print(f"  Test decode: {tokens} -> '{decoded}'")

        return True
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False

def test_model_config():
    """Test Qwen2.5-0.5B config can be loaded"""
    print("\n" + "=" * 50)
    print("Step 3: Testing model config loading")
    print("=" * 50)

    model_path = "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B"

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Config loaded")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Num attention heads: {config.num_attention_heads}")
        print(f"  Intermediate size: {config.intermediate_size}")
        print(f"  Vocab size: {config.vocab_size}")
        print(f"  Max position embeddings: {config.max_position_embedding}")

        return True
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False

def test_simple_forward():
    """Test a simple forward pass with Megatron model"""
    print("\n" + "=" * 50)
    print("Step 4: Testing simple Megatron model forward pass")
    print("=" * 50)

    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        # Create a tiny model for testing
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            pipeline_dtype=torch.float32,
        )

        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=1000,
            max_sequence_length=128,
        )

        print(f"✓ Model created: {type(model).__name__}")
        print(f"  Num parameters: {sum(p.numel() for p in model.parameters())}")

        # Test forward pass
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))

        with torch.no_grad():
            output = model(input_ids)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("Megatron-LM + Qwen2.5-0.5B Integration Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Basic Imports", test_basic_imports()))
    results.append(("Tokenizer Loading", test_tokenizer()))
    results.append(("Model Config", test_model_config()))
    results.append(("Simple Forward", test_simple_forward()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n���� All tests passed! Megatron-LM is ready for integration.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    return 0 if all_passed else 1

h� __name__ == "__main__":
    sys.exit(main())
