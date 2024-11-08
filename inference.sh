#!/bin/bash

# 定义变量
INPUTDIR="/home/jianglei/work/CoSeR/input_img"               # 替换为实际的输入目录路径
OUTDIR="/home/jianglei/work/CoSeR/output_img"                 # 替换为实际的输出目录路径
CONFIG="/home/jianglei/work/CoSeR/configs/CoSeR/inference.yaml"
LOAD_CKPT="/home/jianglei/work/CoSeR/ckpt/state_dict_coser.pt"  # 替换为实际的模型检查点路径
VQGAN_CKPT="/home/jianglei/work/CoSeR/ckpt/vqgan_cfw_00011.ckpt"   # 替换为实际的VQGAN检查点路径

# 执行命令
CUDA_VISIBLE_DEVICES=2 python scripts/inference_tile.py \
    --inputdir "$INPUTDIR" \
    --outdir "$OUTDIR" \
    --config "$CONFIG" \
    --load_ckpt "$LOAD_CKPT" \
    --vqgan_ckpt "$VQGAN_CKPT"
