import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from itertools import combinations


def create_decoder_attn_mask(attn_pair, N):
    mask = torch.full((N, N), float('-inf'))  # 初始化为 -inf
    # 遍历 attn_pair 列表
    for pair in attn_pair:
        # 生成 tuple 中的所有两两组合
        for i, j in combinations(pair, 2):
            # 注意力互相可见
            mask[i, j] = 0
            mask[j, i] = 0
        # 自注意力
        for i in pair:
            mask[i, i] = 0  # 自身注意力
    return mask


class SelfAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=0., proj_drop=0., attn_l2_norm=False)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio))

    def forward(self, x, attn_bias=None):
        x = x + self.attn(self.norm1(x), attn_bias)
        x = x + self.ffn(self.norm2(x))

        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_drop=0., proj_drop=0., attn_l2_norm=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop        
        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, attn_bias):
        B, L, C = x.shape
        
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        # qkv: BL3Hc

        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2               # q or k or v: BHLc
        
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        out = scaled_dot_product_attention(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(out))


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        return self.drop(self.fc2( self.act(self.fc1(x)) ))
