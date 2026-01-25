import torch
import time

def fused_quant_dequant_impl(weight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int, n_bits: int) -> torch.Tensor:
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    
    # Reshape
    w_reshaped = weight.view(out_features, -1, group_size)
    s = scales.view(out_features, -1, 1)
    z = qzeros.view(out_features, -1, 1)
    
    scale_max = float((1 << n_bits) - 1)
    
    # Quantize with STE (Match Baseline Order: round(w/s) + z)
    x = w_reshaped / s
    
    # Round (STE)
    x_round = x.round()
    x_ste = (x_round - x).detach() + x
    
    # Add Zero Point
    w_int_raw = x_ste + z
    
    # Clamp (STE)
    w_int = w_int_raw.clamp(0.0, scale_max)
    w_int = (w_int - w_int_raw).detach() + w_int_raw
    
    # Dequantize
    w_deq = (w_int - z) * s
    
    return w_deq.view(out_features, in_features)

def test_compile():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    device = "cuda"
    out_features = 4096
    in_features = 4096
    group_size = 128
    n_bits = 4
    
    weight = torch.randn(out_features, in_features, device=device)
    scales = torch.ones(out_features, in_features // group_size, 1, device=device)
    qzeros = torch.zeros(out_features, in_features // group_size, 1, device=device)
    
    # Compile
    print("Compiling...")
    try:
        opt_fn = torch.compile(fused_quant_dequant_impl)
        # Warmup
        for _ in range(5):
            _ = opt_fn(weight, scales, qzeros, group_size, n_bits)
        print("Compilation successful.")
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
             _ = opt_fn(weight, scales, qzeros, group_size, n_bits)
        torch.cuda.synchronize()
        print(f"Compiled Time: {(time.time() - start)*10:.3f} us/iter")
        
    except Exception as e:
        print(f"Compilation failed: {e}")

if __name__ == "__main__":
    test_compile()
