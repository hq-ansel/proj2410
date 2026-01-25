import torch
import torch.nn as nn

@torch.jit.script
def fused_quant_dequant_impl(weight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int, n_bits: int) -> torch.Tensor:
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    
    # Reshape
    w_reshaped = weight.view(out_features, -1, group_size)
    s = scales.view(out_features, -1, 1)
    z = qzeros.view(out_features, -1, 1)
    
    scale_max = float((1 << n_bits) - 1)
    
    # Quantize
    w_int = (w_reshaped / s + z).round().clamp(0.0, scale_max)
    
    # Dequantize
    w_deq = (w_int - z) * s
    
    return w_deq.view(out_features, in_features)

def pytorch_quant_dequant(weight, scales, qzeros, group_size, n_bits):
    out_features, in_features = weight.shape
    s = scales.view(out_features, -1, 1)
    z = qzeros.view(out_features, -1, 1)
    w_reshaped = weight.view(out_features, -1, group_size)
    
    # Identical logic in PyTorch
    w_int = (w_reshaped / s + z).round().clamp(0, (1 << n_bits) - 1)
    w_deq = (w_int - z) * s
    return w_deq.view(out_features, in_features)

def test_precision():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    out_features = 128
    in_features = 128
    group_size = 128
    n_bits = 4
    
    # Random large weights to test range
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.float32)
    scales = torch.rand(out_features, 1, device=device, dtype=torch.float32) + 0.1
    qzeros = torch.zeros(out_features, 1, device=device, dtype=torch.float32)
    
    # 1. Float32 Test
    print("Testing Float32...")
    out_jit = fused_quant_dequant_impl(weight, scales, qzeros, group_size, n_bits)
    out_pt = pytorch_quant_dequant(weight, scales, qzeros, group_size, n_bits)
    
    diff = (out_jit - out_pt).abs().max()
    print(f"Float32 Max Diff: {diff:.8e}")
    
    # 2. Check if casting to int32 intermediate in PyTorch causes diff?
    # The JIT implementation keeps w_int as float for clamp/sub, then implicit cast? 
    # Actually my JIT implementation:
    # w_int = (w_reshaped / s + z).round().clamp(0.0, scale_max) -> This is float
    # w_deq = (w_int - z) * s -> This is float calculation
    
    # In `IntQuantLinear.py`:
    # w_int = (w_reshaped / s + z).round().clamp(...)
    # w_int = w_int.to(torch.int32)  <-- THIS CAST might be the key?
    # w_deq = (w_int - z) * s
    
    # Let's modify PyTorch version to match exact logic of IntQuantLinear
    def pytorch_quant_dequant_exact(weight, scales, qzeros, group_size, n_bits):
        out_features, in_features = weight.shape
        s = scales.view(out_features, -1, 1)
        z = qzeros.view(out_features, -1, 1)
        w_reshaped = weight.view(out_features, -1, group_size)
        
        w_int = (w_reshaped / s + z).round().clamp(0, (1 << n_bits) - 1)
        # The cast to int32 in baseline!
        # w_int = w_int.to(torch.int32) 
        # Actually `IntQuantLinear.get_int_weight` does cast. 
        # But `_reference_forward_unpacked` in my Infra implementation (before optimization) 
        # did NOT cast to int32 explicitly in the float path? 
        # Wait, let me check `int_quant_linear.py` forward pass again.
        
        return (w_int - z) * s
        
    # Re-check
    
if __name__ == "__main__":
    test_precision()
