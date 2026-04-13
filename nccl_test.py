#!/usr/bin/env python3
"""
最小 NCCL 跨机通信测试
用法（两台机器各自执行）：
  s53 (rank 0, master):
    NCCL_SOCKET_IFNAME=wg0 NCCL_IB_DISABLE=1 \
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
      --rdzv_endpoint=10.200.0.3:29500 nccl_test.py

  s54 (rank 1):
    NCCL_SOCKET_IFNAME=wg0 NCCL_IB_DISABLE=1 \
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
      --rdzv_endpoint=10.200.0.3:29500 nccl_test.py
"""
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 每个 rank 创建一个值为 rank+1 的 tensor
    t = torch.tensor([float(rank + 1)], device=device)
    print(f"[rank {rank}/{world}] before allreduce: {t.item():.1f}", flush=True)

    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}/{world}] after  allreduce: {t.item():.1f}  "
          f"(expected {sum(range(1, world+1)):.1f})", flush=True)

    # 验证结果正确
    expected = float(sum(range(1, world + 1)))
    assert abs(t.item() - expected) < 1e-3, f"NCCL result wrong: {t.item()} != {expected}"
    if rank == 0:
        print(f"\n✓ NCCL 跨机通信测试通过  world_size={world}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
