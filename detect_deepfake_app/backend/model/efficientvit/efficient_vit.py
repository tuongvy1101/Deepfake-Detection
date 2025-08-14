import torch
from torch import nn
from einops import rearrange
from efficientnet_pytorch import EfficientNet
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class EfficientViT(nn.Module):
    def __init__(self, config, channels=1280, selected_efficient_net=0, checkpoint_path=None):
        super().__init__()
        image_size = config['model']['image-size']
        patch_size = config['model']['patch-size']
        num_classes = config['model']['num-classes']
        dim = config['model']['dim']
        depth = config['model']['depth']
        heads = config['model']['heads']
        mlp_dim = config['model']['mlp-dim']
        emb_dim = config['model']['emb-dim']
        dim_head = config['model']['dim-head']
        dropout = config['model']['dropout']
        emb_dropout = config['model']['emb-dropout']

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selected_efficient_net = selected_efficient_net

        # Khởi tạo EfficientNet
        if selected_efficient_net == 0:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7')

        # Đóng băng các layer đầu của EfficientNet
        for i in range(0, len(self.efficient_net._blocks)):
            for param in self.efficient_net._blocks[i].parameters():
                param.requires_grad = i >= len(self.efficient_net._blocks) - 3

        # Số patch dựa trên kích thước feature map (thường 7x7 sau EfficientNet)
        self.patch_size = patch_size
        num_patches = (7 // patch_size) ** 2  # Giả định feature map 7x7
        patch_dim = channels * patch_size ** 2
        logger.info(f"num_patches: {num_patches}, patch_dim: {patch_dim}")

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # Shape [1, num_patches + 1, dim]
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

        # Tải checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            try:
                self.load_state_dict(checkpoint, strict=False)  # Sử dụng strict=False để bỏ qua lỗi kích thước
                logger.info("Checkpoint loaded successfully with strict=False")
            except RuntimeError as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.info("Using pretrained EfficientNet weights instead")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found. Using pretrained EfficientNet weights.")

        self.to(self.device)

    def forward(self, img, mask=None):
        if img.dim() != 4:
            raise ValueError(f"Expected input shape (batch, channels, height, width), got {img.shape}")

        p = self.patch_size
        x = self.efficient_net.extract_features(img.to(self.device))  # Shape: [batch_size, 1280, 7, 7]
        h, w = x.shape[2], x.shape[3]
        logger.info(f"Feature map shape: {x.shape}, height: {h}, width: {w}")

        # Tính số patch dựa trên kích thước feature map
        expected_patches = (h // p) * (w // p)
        logger.info(f"Expected patches: {expected_patches}")

        # Trích xuất patch
        x = nn.functional.unfold(x, kernel_size=p, stride=p)  # Shape: [batch_size, channels * p * p, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, patch_dim]
        logger.info(f"Unfolded tensor shape: {x.shape}, Sample value: {x[0, 0, :5]}")

        # Đảm bảo số patch khớp với pos_embedding
        num_patches_expected = self.pos_embedding.shape[1] - 1  # Trừ cls_token
        if x.shape[1] != num_patches_expected:
            if x.shape[1] < num_patches_expected:
                padding = torch.zeros(x.shape[0], num_patches_expected - x.shape[1], x.shape[2]).to(self.device)
                x = torch.cat((x, padding), dim=1)
            elif x.shape[1] > num_patches_expected:
                x = x[:, :num_patches_expected, :]
            logger.info(f"Adjusted tensor shape to match pos_embedding: {x.shape}")

        y = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), dim=1)
        x += self.pos_embedding.to(x.device)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)