from engine_for_finetuning import (
    final_test,
    merge,
    merge_predictions,
    train_one_epoch,
    validation_one_epoch,
)
import argparse

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,default='/projects/results/videomaev2/finetune/24.04.30/wlasl_2000/vit_b_32_wlasl_2000_head_hands_merged_from_asl_citizen_ft_/1', help='Path to pred_files')
    parser.add_argument('--num_tasks', type=int, default=1, help='Path to save the modified CSV file')
    args = parser.parse_args()

    #output_dir = args.output_dir
    num_tasks = args.num_tasks
    #merge(output_dir,8)
    output_dir = input("Enter the path to the output directory: ")
    final_top1, final_top5, final_top1_cls, final_top5_cls = merge_predictions(output_dir, num_tasks)
    print(
        f"Accuracy of the network on the test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%, Top-1_cls: {final_top1_cls:.2f}%, Top-5_cls: {final_top5_cls:.2f}%")