import torch
import torch.distributed as dist


def setup(rank, world_size):
    if not dist.is_initialized():  # 检查是否已经初始化
        dist.init_process_group(
            backend="nccl", 
            init_method="tcp://127.0.0.1:29500", 
            rank=rank, 
            world_size=world_size
        )
    torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized successfully.")


def main():
    world_size = torch.cuda.device_count()
    for rank in range(world_size):
        setup(rank, world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    cleanup()  # 进程退出时销毁进程组
