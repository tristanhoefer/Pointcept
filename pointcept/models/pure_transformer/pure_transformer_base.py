import torch
import torch.nn as nn

from pointcept.models.builder import MODELS
import torch
import flash_attn

class PointCloudEmbedding(nn.Module):
    def __init__(self, num_points=1024, in_channels=3, embed_dim=64):
        super(PointCloudEmbedding, self).__init__()
        self.num_points = num_points
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, num_points, embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, enable_flash=False, patch_size_max=16, upcast_attention=False, enable_rpe=False, upcast_softmax=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.enable_flash = enable_flash
        self.patch_size_max = patch_size_max
        self.upcast_attention = upcast_attention
        self.enable_rpe = enable_rpe
        self.upcast_softmax = upcast_softmax

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        K = min(N, self.patch_size_max)
        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if not self.enable_flash:
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(x))
            if self.upcast_softmax:
                attn = attn.float()
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn).to(qkv.dtype)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            cu_seqlens = torch.arange(0, (B + 1) * N, step=N, dtype=torch.int32, device=x.device)
            qkv_packed = qkv.permute(1, 0, 2, 3, 4).reshape(B * N, 3, H, self.head_dim)
            x = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv_packed.half(),
                cu_seqlens,
                max_seqlen=K,
                dropout_p=self.attn_drop.p if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(B, N, C).to(qkv.dtype)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

@MODELS.register_module("Pure-Transformer")
class VisionTransformer(nn.Module):
    def __init__(self, num_points=1024, in_channels=3, num_classes=20, embed_dim=64, depth=6, num_heads=8, mlp_ratio=4.0, drop=0.1):
        super(VisionTransformer, self).__init__()
        self.point_embed = PointCloudEmbedding(num_points, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_points, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.point_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x