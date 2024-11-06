#!/bin/bash

# �������
INPUTDIR="/home/jianglei/work/CoSeR/input_img"               # �滻Ϊʵ�ʵ�����Ŀ¼·��
OUTDIR="/home/jianglei/work/CoSeR/output_img"                 # �滻Ϊʵ�ʵ����Ŀ¼·��
CONFIG="/home/jianglei/work/CoSeR/configs/CoSeR/inference.yaml"
LOAD_CKPT="/home/jianglei/work/CoSeR/ckpt/state_dict_coser.pt"  # �滻Ϊʵ�ʵ�ģ�ͼ���·��
VQGAN_CKPT="/home/jianglei/work/CoSeR/ckpt/vqgan_cfw_00011.ckpt"   # �滻Ϊʵ�ʵ�VQGAN����·��

# ִ������
CUDA_VISIBLE_DEVICES=2 python scripts/inference_tile.py \
    --inputdir "$INPUTDIR" \
    --outdir "$OUTDIR" \
    --config "$CONFIG" \
    --load_ckpt "$LOAD_CKPT" \
    --vqgan_ckpt "$VQGAN_CKPT"
