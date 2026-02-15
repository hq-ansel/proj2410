# ddp_min.py
import os
import socket
import torch
import torch.distributed as dist


def main():
    # 1) init
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 2) bind GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(
            f"[rank {rank}/{world}] host={socket.gethostname()} "
            f"MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
            flush=True,
        )
    print(
        f"[rank {rank}/{world}] local_rank={local_rank} cuda={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(device)}",
        flush=True,
    )

    # 3) a tiny collective
    x = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] all_reduce result = {x.item()} (expected {world*(world+1)/2})", flush=True)

    # 4) finish
    dist.barrier()
    if rank == 0:
        print("done.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
