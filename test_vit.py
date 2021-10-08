import torch
from vit_pytorch import ViT

if __name__ == '__main__':
    v = ViT(
    image_size = 768,
    patch_size = 24,
    channels = 1,
    num_classes = 256,
    dim = 1024,
    depth = 5,
    heads = 16,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
    )
    img = torch.randn(1, 1,768,384)
    preds = v(img)
    print(preds)

