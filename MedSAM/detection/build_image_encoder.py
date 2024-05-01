from functools import partial
import torch
import urllib.request
from pathlib import Path
from detection.image_encoder import ImageEncoderViT
import torch.optim as optim


def download_checkpoint(model_name, checkpoint_path):
    checkpoint_url = {
        'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    if model_name in checkpoint_url:
        checkpoint_url = checkpoint_url[model_name]
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {model_name} checkpoint...")
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        print(f"{model_name} checkpoint downloaded!")
    else:
        raise ValueError(f"Checkpoint URL for model '{model_name}' not found.")


def build_MedSAM_image_encoder(model_name='vit_b', checkpoint=None):
    encoder_config = {
        'vit_h': (1280, 32, 16, [7, 15, 23, 31]),
        'vit_l': (1024, 24, 16, [5, 11, 17, 23]),
        'vit_b': (768, 12, 12, [2, 5, 8, 11]),
    }
    if model_name not in encoder_config:
        raise ValueError(f"Model '{model_name}' not supported.")

    encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes = encoder_config[model_name]

    image_size = 1024
    vit_patch_size = 16

    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=256,
    )
    image_encoder.eval()

    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            download_checkpoint(model_name, checkpoint_path)
        print(f'load checkpoint')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # print("Checkpoint keys:", checkpoint.keys())

        image_encoder_state_dict = {
            key.replace('image_encoder.', ''): value
            for key, value in checkpoint.items()
            if key.startswith('image_encoder.')
        }
        
        image_encoder.load_state_dict(image_encoder_state_dict)

    return image_encoder
