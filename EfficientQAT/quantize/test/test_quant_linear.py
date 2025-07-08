import torch
import numpy as np
from typing import Optional
import pytest
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init

import math

# from ..int_linear_fake import QuantLinear as QuantLinearFake
# from ..int_linear_real import QuantLinear as QuantLinearReal
from EfficientQAT.quantize.int_linear_fake import QuantLinear as QuantLinearFake
from EfficientQAT.quantize.int_linear_real import QuantLinear as QuantLinearReal

@torch.no_grad()
def init_fake_real_linear( linear:Linear):
    args= { }
    dtype = linear.weight.dtype
    linear_q_fake = QuantLinearFake(
        linear,
        wbits=2,
        group_size=128,
        args=args,
    )
    linear_q_fake.set_quant_state(True)
    # quant inplace 
    linear_q_fake.weight.data = linear_q_fake.weight_quantizer(
        linear_q_fake.weight.data
    )
    scales = linear_q_fake.weight_quantizer.scale.clamp(1e-4,1e4).clone()
    zeros = linear_q_fake.weight_quantizer.zero_point.clone()
    group_size = linear_q_fake.weight_quantizer.group_size
    dim0 = linear_q_fake.weight.shape[0]
    scales = scales.view(dim0,-1).transpose(0,1).contiguous()
    zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
    # import pdb;pdb.set_trace()
    linear_q_real = QuantLinearReal(
        bits= 2,
        group_size=128,
        infeatures=linear.in_features,
        outfeatures=linear.out_features,
        bias=None,
    )
    linear_q_real.pack(linear_q_fake.cpu(), scales.cpu(), zeros.cpu())
    linear_q_fake.cuda(), linear_q_real.cuda()
    
    f_w = linear_q_fake.weight.data
    q_f_w = linear_q_fake.weight_quantizer(f_w)

    r_w = linear_q_real.get_weight(transpose=True,dtype=dtype)
    # torch.cuda.synchronize()
    # r_w2 = linear_q_real.get_weight(transpose=True,dtype=dtype)
    # # float16能够表达的最小正数是5.9e-8
    # assert torch.allclose(r_w, r_w2, atol=1e-9)
    
    assert f_w.dtype == q_f_w.dtype == r_w.dtype == dtype , f"{f_w.dtype} {q_f_w.dtype} {r_w.dtype} {dtype}"
    
    assert torch.allclose(f_w, q_f_w, atol=1e-7)
    assert torch.allclose(q_f_w, r_w, atol=1e-7)
    assert torch.allclose(f_w, r_w, atol=1e-7)
    
    # print(f"real and fake diff {torch.sum(torch.abs(f_w - r_w)>1e-9)}")
    # print(f"fake and q_fake diff {torch.sum(torch.abs(q_f_w - f_w)>1e-9)}")

    return linear_q_fake, linear_q_real


@pytest.mark.parametrize("SIZE", [(4096, 14336),(4096,4096),(4096,11008),
                                  (896,4864),(896,896),(1536,8960),(1536,1536),
                                  (2048,11008),(2048,2048)]) # (in dim, out dim)
@pytest.mark.parametrize("BATCH", [2, 4])
@pytest.mark.parametrize("SEQLEN", [1024, 2048])
@pytest.mark.parametrize("DTYPE", [torch.float16])
@torch.no_grad()
def test_linear(SIZE, BATCH, SEQLEN,  DTYPE):
    if DTYPE == torch.bfloat16:
        # TODO should match bfloat16 gemm with pytorch.
        rtol, atol = (1e-2, 3e-3)
    else:
        rtol, atol = (1e-3, 3e-3)

    rawlinear = Linear(SIZE[0], SIZE[1], False,dtype=DTYPE).cuda()
    # 原始的linear.weight 的形状为(out_features, in_features)
    init.kaiming_uniform_(rawlinear.weight, a=math.sqrt(5))
    # input_tensor = torch.randn([BATCH, SEQLEN, SIZE[0]], dtype=DTYPE, device="cuda")
    # 全0测试 pass
    # input_tensor = torch.zeros([BATCH, SEQLEN, SIZE[0]], dtype=DTYPE, device="cuda")
    # 全1测试 pass
    # input_tensor = torch.ones([BATCH, SEQLEN, SIZE[0]], dtype=DTYPE, device="cuda")
    # 单位矩阵 pass
    # input_tensor = torch.eye(SIZE[0], dtype=DTYPE, device="cuda")
    # 渐变输入
    input_tensor = torch.linspace(-5.5, 5.7, steps=BATCH * SEQLEN * SIZE[0],
         dtype=DTYPE, device="cuda").view(BATCH, SEQLEN, SIZE[0]).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    print(f"raw linear weight max {torch.max(rawlinear.weight)} min {torch.min(torch.abs(rawlinear.weight))}")
    print(f"input max {torch.max(input_tensor)} min {torch.min(input_tensor)}")

    linear_q_fake, linear_q_real = init_fake_real_linear(rawlinear)

    tensor_fake = linear_q_fake.forward(input_tensor)
    tensor_real = linear_q_real.forward(input_tensor)

    # linear_q_real.use_fake_quantization()
    # tensor_us_fake = linear_q_real.forward(input_tensor)
    
    assert linear_q_fake.bias is None
    assert linear_q_real.bias is None

    assert (DTYPE == tensor_fake.dtype 
        == tensor_real.dtype 
        ),f"{DTYPE} {tensor_fake.dtype} {tensor_real.dtype} "
    atol = 1e-5
    print(f"amax = {torch.amax(tensor_real - tensor_fake)} norm = {torch.norm(tensor_real - tensor_fake)}")
    # 统计相差大于atol的个数
    print("real == fake?",torch.sum(torch.abs(tensor_real - tensor_fake) > atol))
    # print("us fake == fake?",torch.sum(torch.abs(tensor_us_fake - tensor_fake) > atol))


    assert torch.allclose(tensor_fake, tensor_real,atol=atol)

# export PYTHONPATH=$PYTHONPATH:/home/ubuntu/data/exp/proj2410
if __name__ == "__main__":
    # test_linear((4096, 14336), 4, 2,  torch.bfloat16)
    # test_linear((896, 896), 4, 2048,  torch.bfloat16)
    # test_linear((4096, 14336), 2, 1024,  torch.float32)
    # test_linear((4096, 4096), 4, 2048,  torch.bfloat16)
    test_linear((4096, 14336), 4, 2048,  torch.float16)
    test_linear((4096, 4096), 2, 1024,  torch.float16)