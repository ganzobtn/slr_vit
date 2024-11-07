

To fine-tune VideoMAEv2 ViT-b on ASL-Citizen with 4 A100-40G (1 nodes x 4 GPUs), you can use the following script file **scripts/finetune/distribute/wlasl_2000/vit_b_32_wlasl_2000_ft_dgx_from_asl_citizen.sh**.


Start training by running
```bash
bash scripts/finetune/distribute/wlasl_2000/vit_b_32/dgx/vit_b_32_wlasl_2000_head_hands_merged_ft_dgx_from_asl_citizen.sh wlasl_2000_vit_b_32 
```
, where wlasl_2000_vit_b_32 is the job name.

If you just want to **test the performance of the model**, change `MODEL_PATH` to the model to be tested, `OUTPUT_DIR` to the path of the folder where the test results are saved, and run the following command:
```bash
bash scripts/finetune/distribute/wlasl_2000/vit_b_32/dgx/vit_b_32_wlasl_2000_head_hands_merged_ft_dgx_from_asl_citizen.sh wlasl_2000_vit_b_32_test --eval
