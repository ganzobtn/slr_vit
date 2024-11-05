#!/usr/bin/env bash
set -x


export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes


#OUTPUT_DIR='YOUR_PATH/work_dir/vit_b_hybrid_pt_800e_k400_ft'
#DATA_PATH='YOUR_PATH/data/k400'
#MODEL_PATH='YOUR_PATH/model_zoo/vit_b_hybrid_pt_800e.pth'


# finetune data list file follows the following format
# for the video data line: video_path, label
# for the rawframe data line: frame_folder_path, total_frames, label

#OUTPUT_DIR='/media/ganzorig/53D11DCE629A37AA/results/videomaev2/vit_b_hybrid_pt_800e_k400_ft'  # Your output folder for deepspeed config file, logs and checkpoints
#DATA_PATH='/home/ganzorig/docker_workspace/videomaev2/'
#MODEL_PATH='/home/ganzorig/docker_workspace/videomaev2/model_zoo/vit_g_hybrid_pt_1200e.pth'  # Model for initializing parameters

OUTPUT_DIR='/l/users/ganzorig.batnasan/results/videomaev2/finetune/vit_b_32_wlasl_1000_from_asl_citizen_ft/0'
DATA_ROOT='/tmp/data/wlasl_2000/WLASL2000_head_hands_merged'
DATA_PATH='/home/ganzorig.batnasan/videomaev2/datas/dgx/finetune/revised/24.04.27/wlasl_1000/'
MODEL_PATH='/l/users/ganzorig.batnasan/results/videomaev2/finetune/vit_b_32_asl_citizen_head_hands_merged_ft/0/checkpoint-best/mp_rank_00_model_states.pt'
N_NODES=2
GPUS_PER_NODE=4 # ${GPUS_PER_NODE:-8}  # Number of GPUs in each node
#PY_ARGS=${@:3}  # Other training args


# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
#OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc_per_node=2 \

OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc_per_node=${GPUS_PER_NODE} \
        --node_rank=1 \
        --master_addr=192.168.52.114 --master_port=12320 \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set WLASL-2000 \
        --nb_classes 1000 \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 1 \
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
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval 
         #--enable_deepspeed \
#        ${PY_ARGS}