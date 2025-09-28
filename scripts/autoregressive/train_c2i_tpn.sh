# !/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0 torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
 --master_port=12345 \
autoregressive/train/train_c2i_tpn.py "$@"
