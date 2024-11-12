from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F
import math
import spconv.pytorch as spconv

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointSequential, PointModule

import torch
import flash_attn


'''
From R. Wightman
https://github.com/rwightman/pytorch-image-models
'''

import torch
import torch.nn as nn
from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 1, 3, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        """
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        """


        if q.dtype != torch.float16:
            q = q.to(torch.float16)
        if k.dtype != torch.float16:
            k = k.to(torch.float16)
        if v.dtype != torch.float16:
            v = v.to(torch.float16)
        try:

            x = flash_attn.flash_attn_func(q,k,v)
            #x = flash_attn.flash_attn_qkvpacked_func(qkv)

        except RuntimeError:
            print("q.shape", q.shape)
            print("k.shape", k.shape)
            print("v.shape", v.shape)
            print("q datatype", q.dtype)
            print("k datatype", k.dtype)
            print("v datatype", v.dtype)
            print("q device", q.device)
            print("k device", k.device)
            print("v device", v.device)
        
        x = x.to(qkv.dtype).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    # Modified Block by adding LayerScale
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, init_values=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y = self.attn(self.norm1(x), mask)
        y = self.ls1(y)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x



class Embedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, norm_layer=None, act_layer=None):
        super(Embedding, self).__init__()
        
        self.proj = nn.Sequential()
        self.proj.append(nn.Linear(in_channels, embed_dim))

        if norm_layer is not None:
            self.proj.append(norm_layer(embed_dim))
        if act_layer is not None:
            self.proj.append(act_layer())

    def forward(self, input):
        input = self.proj(input)  # (num_points, embed_dim)
        return input

        
"""
class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = nn.Sequential()
        self.stem.append(spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ))
        
        if norm_layer is not None:
            self.stem.append(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.append(act_layer(), name="act")

        
        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")
        
            
    def forward(self, point: Point):
        point = self.stem(point)
        return point
"""

@MODELS.register_module("PureTransformer-Base")
class PureTransformerV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20, embed_dim=64, depth=6, num_heads=8, mlp_ratio=4, drop=0.1, drop_path=0.3):
        super(PureTransformerV3, self).__init__()


        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels, embed_dim, norm_layer=bn_layer, act_layer=act_layer)
        #self.embedding = Embedding(in_channels, embed_dim)
        #self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_proj_layer = nn.Linear(3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(p=drop)
        self.embed_dim = embed_dim

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop, drop_path)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.drop_path = PointSequential(
            DropPath(0.3) if 0.3 > 0.0 else nn.Identity() #use variable for drop path
        )

    def forward(self, x):
        point = Point(x) #features are (num_points, 6)
        features = point.feat # [num_points, 6]
        features = self.embedding(features) # [num_points, embed_dim]

        #coords = point.coord # [num_points, 3]
        #pos_encoding = self.pos_proj_layer(coords) # [num_points, embed_dim]
        #features = features + pos_encoding # Shape?
            
        features = self.dropout(features)

        batch_encoding =  point.batch
        # And `batch` is an array of shape [num_points] with values from 0 to batch_size - 1
        batch_size = batch_encoding.max().item() + 1 

        max_points_per_batch = max((batch_encoding == b).sum().item() for b in range(batch_size))

        # Use -1 for num_points to handle varying number of points per batch
        batched_features = torch.zeros((batch_size, max_points_per_batch, self.embed_dim), dtype=features.dtype, device=features.device)

        # Iterate over each batch index to extract and store the corresponding points
        for b in range(batch_size):
            indices = (batch_encoding == b).nonzero(as_tuple=True)[0]
            batched_features[b, :len(indices), :] = features[indices]

        #if batch_size==1:
        #    batched_features = batched_features.unsqueeze(0)
            
        for blk in self.blocks:
            batched_features = blk(batched_features)

        batched_features = self.norm(features)

        point.feat = batched_features

        return point