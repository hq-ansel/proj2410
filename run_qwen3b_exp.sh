#!/bin/bash
# Qwen2.5-3B 消融实验启动脚本 (3 GPU配置)

set -euo pipefail

# 设置GPU环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
export NPROC_PER_NODE=3
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# ⚠️ 添加EfficientQAT到Python路径
export PYTHONPATH="/home/ubuntu/data/exp/proj2410/EfficientQAT:${PYTHONPATH:-}"

# 预编译CUDA扩展（避免多进程锁竞争）
echo "预编译CUDA扩展..."
python -c "from EfficientQAT.core.quantizer.kernel.fake_quant import fake_quant_ste" 2>/dev/null || true

echo "========================================="
echo "GPU配置: 3张卡 (RTX A6000 x2, RTX 5090 x1)"
echo "模型: Qwen2.5-3B"
echo "方法: Seq2Bit + GJSD"
echo "========================================="

# 运行训练
torchrun \
  --nnodes=1 \
  --nproc-per-node=3 \
  --standalone \
  VeOmni/tasks/quantize/train.py \
  VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-seq2bit-kd-gjsd-3gpus.yaml \
  2>&1 | tee logs/qwen3b_seq2bit_gjsd_$(date +%Y%m%d_%H%M%S).log

echo "训练完成！"
