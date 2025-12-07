import torch
import torch.nn as nn
import torch.nn.functional as F

# config tokens block slide 
t=32 # tokens
blk_len = 8 # blocks 为了计算的考虑
slide_stride = 8 # 压缩词元组的最大单元
num_blocks = (t-blk_len)//slide_stride + 1 # 块的数量
# slide_stride<blk_len


dim = 16
heads = 4
head_dim = dim // heads
batch_size = 1

hidden_states = torch.randn(batch_size, t, dim)

Wq,Wk,Wv = torch.rand(dim,dim),torch.rand(dim,dim),torch.rand(dim,dim)

# Q,K,V (bs,t,dim)
Q,K,V = hidden_states@Wq,hidden_states@Wk,hidden_states@Wv

# 降秩算子
W_K_cmp = torch.rand(blk_len,1)
W_V_cmp = torch.rand(blk_len,1)
# position embedding ? 这个怎么加入进来呢？
# 正常顺序是  计算完 K V 后，再加上 position embedding 再求spda
W_pe = torch.randn(blk_len, dim)
# Step 1: 分块操作 [bs, t, dim] -> [bs, num_blocks, blk_len, dim]
# Step 2: 降秩操作 [bs, num_blocks, blk_len, dim] -> [bs, num_blocks, dim]
k_cmp = (K.unfold(1, blk_len, slide_stride).transpose(2,3)@W_K_cmp).transpose(1,2)
v_cmp = (V.unfold(1, blk_len, slide_stride).transpose(2,3)@W_V_cmp).transpose(1,2)
# k_cmp, v_cmp shape (bs,dim,num_blocks)


score_cmp = (Q@k_cmp)
# score_cmp shape (bs,t,num_blocks)
p_cmp = F.softmax(score_cmp, dim=-1)

slc_blk_size = 4
