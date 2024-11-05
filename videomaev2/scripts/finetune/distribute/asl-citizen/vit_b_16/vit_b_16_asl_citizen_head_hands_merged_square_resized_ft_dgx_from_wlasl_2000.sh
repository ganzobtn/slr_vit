#!/usr/bin/env bash
set -x


export MASTER_PORT=${MASTER_PORT:-12321}  # You should set the same master_port in all the nodes


#OUTPUT_DIR='/projects/results/videomaev2/finetune/24.0/aslcitizen_2731/vit_b_16_asl_citizen_head_hands_merged_resizedft_from_wlasl_2000/0'
DATA_ROOT='/projects/data/asl-citizen/ASL_Citizen/ASLCitizen_head_hands_merged_resized'
DATA_PATH='/projects/videomaev2/datas/dgx/finetune/revised/24.04.27/aslcitizen_2731'
MODEL_PATH='/projects/results/videomaev2/finetune/vit_b_wlasl_2000_head_hands_merged_ft/0/checkpoint-best/mp_rank_00_model_states.pt'

N_NODES=1
GPUS_PER_NODE=8 # ${GPUS_PER_NODE:-8}  # Number of GPUs in each node
PY_ARGS=${@:3}  # Other training args

# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT}  \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set WLASL-2000 \
        --nb_classes 2731 \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
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
        --dist_eval  --enable_deepspeed \
        ${PY_ARGS}