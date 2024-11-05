#!/usr/bin/env bash
set -x


export MASTER_PORT=${MASTER_PORT:-12321}  # You should set the same master_port in all the nodes

#OUTPUT_DIR='/l/users/ganzorig.batnasan/results/videomaev2/finetune/24.04.27/wlasl_1573/vit_b_32_wlasl_1573_head_hands_merged_ft_from_asl/0'
OUTPUT_DIR='/projects/results/videomaev2/finetune/test/wlasl_boston/vit_b_32_wlasl_1573_head_hands_merged_ft_dgx_from_asl_citizen/1'

# DATA_ROOT='/tmp/data/wlasl_2000/WLASL2000_head_hands_merged/'
DATA_ROOT='/projects/data/wlasl_2000/WLASL2000_head_hands_merged/'

# DATA_PATH='/home/ganzorig.batnasan/videomaev2/datas/cscc/finetune/revised/24.04.27/wlasl_1573/'
DATA_PATH='/projects/videomaev2/datas/cscc/finetune/revised/24.04.27/wlasl_1573/'

#MODEL_PATH='/l/users/ganzorig.batnasan/results/videomaev2/finetune/vit_b_32_asl_citizen_head_hands_merged_ft/0/checkpoint-best/mp_rank_00_model_states.pt'
MODEL_PATH='/projects/results/videomaev2/finetune/24.05.07/wlasl_1573/vit_b_32_wlasl_1573_head_hands_merged_ft_from_asl/0/checkpoint-best/mp_rank_00_model_states.pt'

N_NODES=1
GPUS_PER_NODE=1 # ${GPUS_PER_NODE:-8}  # Number of GPUs in each node
PY_ARGS=${@:3}  # Other training args

# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="1"  python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT}  \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set WLASL-2000 \
        --nb_classes 1573 \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 32 \
        --sampling_rate 2 \
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
        --sparse_sample \
        --test_num_segment 1 \
        --test_num_crop 1 \
        --eval \
        --enable_deepspeed 
        #${PY_ARGS}
  
