#!/usr/bin/env bash
set -x


export MASTER_PORT=${MASTER_PORT:-12321}  # You should set the same master_port in all the nodes


#OUTPUT_DIR='YOUR_PATH/work_dir/vit_b_hybrid_pt_800e_k400_ft'
#DATA_PATH='YOUR_PATH/data/k400'
#MODEL_PATH='YOUR_PATH/model_zoo/vit_b_hybrid_pt_800e.pth'


# finetune data list file follows the following format
# for the video data line: video_path, label
# for the rawframe data line: frame_folder_path, total_frames, label

#OUTPUT_DIR='/media/ganzorig/53D11DCE629A37AA/results/videomaev2/vit_b_hybrid_pt_800e_k400_ft'  # Your output folder for deepspeed config file, logs and checkpoints
#DATA_PATH='/home/ganzorig/docker_workspace/videomaev2/'
#MODEL_PATH='/home/ganzorig/docker_workspace/videomaev2/model_zoo/vit_g_hybrid_pt_1200e.pth'  # Model for initializing parameters

OUTPUT_DIR='/projects/results/videomaev2/finetune/vit_b_32_wlasl_1000_ft/0'
DATA_ROOT='/projects/data/wlasl_2000'
DATA_PATH='/projects/videomaev2/datas/dgx/finetune/revised/24.04.27/wlasl_1000/'
#MODEL_PATH='/projects/libraries/pretrained_models/videomaev2/vit_b_k710_dl_from_giant.pth'
MODEL_PATH='/projects/results/videomaev2/finetune/24.02.23/aslcitizen_2731/vit_b_32_asl_citizen_ft/0/checkpoint-best/mp_rank_00_model_states.pt'
N_NODES=1
GPUS_PER_NODE=4 # ${GPUS_PER_NODE:-8}  # Number of GPUs in each node
PY_ARGS=${@:3}  # Other training args



# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT}  \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set WLASL-2000 \
        --nb_classes 1000 \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
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
        --epochs 200 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}