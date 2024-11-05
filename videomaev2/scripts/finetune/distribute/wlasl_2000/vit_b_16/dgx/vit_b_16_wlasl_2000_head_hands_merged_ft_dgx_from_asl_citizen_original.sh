#!/usr/bin/env bash
set -x


export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes

OUTPUT_DIR='/projects/results/videomaev2/finetune/24.05.15/wlasl_2000/vit_b_16_wlasl_2000_head_hands_merged_from_asl_citizen_original_ft/0'
DATA_PATH='/projects/videomaev2/datas/dgx/finetune/revised/WLASL2000_head_hands_merged'
MODEL_PATH='/projects/results/videomaev2/finetune/24.02.23/aslcitizen_2731/vit_b_16_asl_citizen_ft/0/checkpoint-best/mp_rank_00_model_states.pt'
N_NODES=1
GPUS_PER_NODE=4 # ${GPUS_PER_NODE:-8}  # Number of GPUs in each node
PY_ARGS=${@:3}  # Other training args

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="4,5,6,7" torchrun --nnodes=1 --nproc_per_node=${GPUS_PER_NODE} \
        --master_port=12320\
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set WLASL-2000 \
        --nb_classes 2000 \
        --data_path ${DATA_PATH}\
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 1 \
        --num_workers 8 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 5 \
        --epochs 50 \
        --test_num_segment 2 \
        --test_num_crop 2 \
        --dist_eval \
        --enable_deepspeed \
        ${PY_ARGS}

	#--dist_eval \
