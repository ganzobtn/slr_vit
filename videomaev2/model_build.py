import torch
from timm.models import create_model
import deepspeed
import utils

model = create_model(
    'vit_base_patch16_224',
    img_size=(240,320),
    pretrained=False,
    num_classes=2000,
    drop_rate=0.4,
    drop_path_rate=0.4,
    attn_drop_rate=0.4,
    drop_block_rate=None,
)
# model = create_model(
#     'vit_base_patch16_224',
#     img_size=224,
#     pretrained=False,
#     num_classes=2000,
#     all_frames=16,
#     tubelet_size=2,
#     drop_rate=0.4,
#     drop_path_rate=0.4,
#     attn_drop_rate=0.4,
#     head_drop_rate=0.4,
#     drop_block_rate=None,
#     use_mean_pooling=True,
#     init_scale=0.001,
#     with_cp=False,
# )
print(model)
x  = torch.randn(1, 3, 240, 320)
y = model(x)
print(y.shape)
checkpoint = torch.load('checkpoints/asl_citizen_2731/vit_b_32_asl_citizen_ft/0/checkpoint-best/mp_rank_00_model_states.pt', map_location='cpu')

#checkpoint_model = 'checkpoints/asl_citizen_2731/vit_b_32_asl_citizen_ft/0/checkpoint-best/mp_rank_00_model_states.pt'
# utils.load_state_dict(model, checkpoint_model)