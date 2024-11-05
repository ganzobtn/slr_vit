import torch
import utils
import math
import os
import sys
from multiprocessing import Pool
from typing import Iterable, Optional
from dataset import video_transforms, volume_transforms
from torchvision import transforms
import numpy as np
import torch
from timm.utils import ModelEma, accuracy
import PIL
import imageio
import cv2
def get_attention_visualize(batch_tensor,attn_map):
    # batch_tensor: B, C, T, H, W
    # attn_map : B, n_heads, n_tokens, n_tokens
    B, C, T, H, W = batch_tensor.shape
    B, n_heads, n_tokens, _ = attn_map.shape

    # get the attention map
    attn_map = attn_map.detach().cpu().numpy()



@torch.no_grad()
def inference_visualize(data_loader, model, device, video_path):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualize:'

    # switch to evaluation mode
    model.eval()

    reverse_transform = video_transforms.Compose([
        video_transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]),
                # volume_transforms.ClipToPILImage()
    ])


    correct = 0
    total = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        print('ids:',ids)
        chunk_nb = batch[3]
        split_nb = batch[4]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        #sprint(ids,chunk_nb,split_nb)

        # compute output
    #     with torch.cuda.amp.autocast():
    #         output = model(images)
    #         # reverse the data transformation
    #         # self.data_transform = video_transforms.Compose([
    #         #     volume_transforms.ClipToTensor(),
    #         #     video_transforms.Normalize(
    #         #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #         # ])
    #         #print(images.shape, torch.is_tensor(images))
    #         attn_map = model.blocks[-1].attn.attn_map
    #         print('attn_map:',attn_map.shape)

    #         C, T, H, W = images.shape[1:]
    #         reversed_videos = torch.empty((0, C, T, H, W), device=device)
    #         for image in images:
    #             #reversed_video = image
    #             reversed_video = reverse_transform(image) 
    #             #print('max:',reversed_video.max())
    #             reversed_videos = torch.cat((reversed_videos, reversed_video.unsqueeze(0)*255), dim=0)
    #         #print('reversed:',reversed_videos.shape)
    #         output_argmax = torch.argmax(output, dim=1)
    #         for i in range(output_argmax.shape[0]):
    #             if target[i] == output_argmax[i]:
    #                 #print('correct '+str(target[i].item())+' '+ids[i])
    #                 correct+=1
    #             total+=1

    #         #print(output_argmax, target)
    #         # save reversed videos to mp4
    #         for batch_num in range(attn_map.shape[0]):

    #             print('attn:',attn_map[i].shape)
    #             for head in range(attn_map.shape[1]):
    #                 patch_size = 16
    #                 temporal_patch = T//2
    #                 height_patch = H//patch_size
    #                 width_patch = W//patch_size
    #                 attn = attn_map[batch_num,head].mean(dim=0)
    #                 # attn = torch.zeros(temporal_patch*height_patch*width_patch).to(device)
    #                 # for ww in range(temporal_patch):
    #                 #     patches_per_frame = height_patch*width_patch
    #                 #     # for k in range(attn_map.shape[2]//T*2):
    #                 #     for k in range(patches_per_frame):
    #                 #         ##print(i,k,attn.shape,attn_map.shape)
    #                 #         #print(attn_map[batch_num,head,i*196:i*196+196,i*196+k].mean())
    #                 #         attn[ww*patches_per_frame+k]= attn_map[batch_num,head,ww*patches_per_frame:ww*patches_per_frame+patches_per_frame,ww*patches_per_frame+k].mean()
    #                 #     attn[ww*patches_per_frame:ww*patches_per_frame+patches_per_frame] /= torch.max(attn[ww*patches_per_frame:ww*patches_per_frame+patches_per_frame])
                        
    #                 attn = attn.reshape(temporal_patch,height_patch,width_patch).cpu().numpy() # 16x14x14
    #                 attn =attn/np.max(attn)

    #                 heatmap_frames = []
    #                 for t in range(attn.shape[0]):
    #                     heatmap = cv2.applyColorMap((attn[t] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #                     heatmap_frames.append(heatmap)
    #                 heatmap_path = os.path.join(video_path, ids[i] + f'_chunk_{chunk_nb[i]}_split_{split_nb[i]}' + '_heatmap_head_'+str(head)+'.mp4')
    #                 heatmap_writer = cv2.VideoWriter(heatmap_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (attn.shape[-1], attn.shape[-2]))
    #                 for frame in heatmap_frames:
    #                     heatmap_writer.write(frame)
    #                 heatmap_writer.release()

    #             # -------------------------------------
    #             # attn= attn_map[i].mean(dim=1)
    #             # attn = attn.reshape(T//2,14,14).cpu().numpy()
    #             # attn =attn/np.max(attn)                
    #             # heatmap_frames = []
    #             # for t in range(attn.shape[0]):
    #             #     heatmap = cv2.applyColorMap((attn[t] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #             #     heatmap_frames.append(heatmap)
    #             # heatmap_path = os.path.join(video_path, ids[i] + f'_chunk_{chunk_nb[i]}_split_{split_nb[i]}' + '_heatmap_dim_1.mp4')
    #             # heatmap_writer = cv2.VideoWriter(heatmap_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (attn.shape[-1], attn.shape[-2]))
    #             # for frame in heatmap_frames:
    #             #     heatmap_writer.write(frame)
    #             # heatmap_writer.release()
                
    #             video_frames = []
    #             for t in range(reversed_videos.shape[2]):
    #                 image = reversed_videos[i, :, t].permute(1, 2, 0).cpu().numpy()
    #                 image = image.astype(np.uint8)
    #                 frame_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #                 #video_frames.append(PIL.Image.fromarray(image))
    #                 video_frames.append(frame_rgb)
    #             #video_path = os.path.join(video_path, f"video_{i}.mp4")
    #             #video_frames[0].save(video_path, format='MP4', append_images=video_frames[1:], save_all=True, fps=30)
                
    #             filename = os.path.join(video_path,ids[i]+f'_chunk_{chunk_nb[i]}_split_{split_nb[i]}'+'.mp4')


    #             # Create a video writer object
    #             writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (reversed_videos.shape[-1], reversed_videos.shape[-2]))


    #             # Write each frame to the video
    #             for frame in video_frames:

    #                 writer.write(frame)


    #             # Close the video writer object
    #             writer.release()


    #             #print('saved')
    #         # save batch tensor to images
    #         # for i in range(reversed_videos.shape[0]):
    #         #     for t in range(reversed_videos.shape[2]):
    #         #         image = reversed_videos[i, :, t].permute(1, 2, 0).numpy()
    #         #         image = image.astype(np.uint8)
    #         #         image_path = os.path.join(video_path, f"image_{i}_{t}.jpg")
    #         #         PIL.Image.fromarray(image).save(image_path)
    #         # for i in range(images.shape[0]):
    #         #     filename = ids[i]+f'_chunk_{chunk_nb[i]}_split_{split_nb[i]}'+'.npy'
    #         #     np.save(os.path.join(video_path,filename),attn_map[i].cpu().numpy())
    # print('correct:',correct,'total:',total,'accuracy:',correct/total)
    #metric_logger.synchronize_between_processes()

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
