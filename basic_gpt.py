import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

# class Independent_Projection(nn.Module):
#     def __init__(self, num_positions, input_dim, hidden_dim):
#         super().__init__()
#         self.num_positions = num_positions
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         self.projections = nn.ModuleList([
#             nn.Linear(input_dim, hidden_dim) for _ in range(num_positions)
#         ])

#     def forward(self, x):
#         """
#         x: [batch_size, num_positions, input_dim]
#         """
#         outputs = []
#         for pos in range(self.num_positions):
#             pos_projection = self.projections[pos](x[:, pos, :])
#             outputs.append(pos_projection)

#         return torch.stack(outputs, dim=1)  # [batch_size, num_positions, hidden_dim]

class Independent_Projection(nn.Module):
    def __init__(self, num_positions, input_dim, hidden_dim):
        super().__init__()
        self.num_positions = num_positions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 随机初始化位置嵌入的权重和偏置
        self.weight = nn.Parameter(torch.randn(num_positions, hidden_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(num_positions, hidden_dim))

    def forward(self, x, input_pos=None):
        """
        x: [batch_size, num_positions, input_dim]
        """
        seq_len = x.shape[1]
        if input_pos is not None:
            # print(self.weight.shape, input_pos*seq_len, (input_pos+1)*seq_len)
            # raise NotImplementedError
            weight = self.weight[input_pos*seq_len:(input_pos+1)*seq_len]
            bias = self.bias[input_pos*seq_len:(input_pos+1)*seq_len]
            # print(weight.shape, x.shape)

            output = torch.einsum('bij,ijk->bik', x, weight.transpose(1, 2)) + bias
        else:
            output = torch.einsum('bij,ijk->bik', x, self.weight[:seq_len].transpose(1, 2)) + self.bias[:seq_len]
        return output  # [batch_size, num_positions, hidden_dim]

    def load_from_linear(self, weight, bias=None):
        """
        从一个现有的 nn.Linear 加载权重到所有位置。
        linear: nn.Linear
        """
        with torch.no_grad():
            # 将 nn.Linear 的权重和偏置加载到每个位置
            self.weight.data.copy_(
                weight.unsqueeze(0).expand(self.num_positions, -1, -1)
            )
            if bias is not None:
                self.bias.data.copy_(
                    bias.unsqueeze(0).expand(self.num_positions, -1)
                )



# class Independent_Projection(nn.Module):
#     def __init__(self, num_positions, input_dim, hidden_dim):
#         super().__init__()
#         self.num_positions = num_positions
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         self.weight_embeddings = nn.Embedding(num_positions, input_dim * hidden_dim)

#     def forward(self, x):
#         """
#         x: [batch_size, num_positions, input_dim]
#         """
#         batch_size, num_positions, input_dim = x.shape

#         position_ids = torch.arange(num_positions, device=x.device).unsqueeze(0).expand(batch_size, num_positions)
#         weight = self.weight_embeddings(position_ids)  # [batch_size, num_positions, input_dim * hidden_dim]

#         weight = weight.view(batch_size, num_positions, input_dim, -1)  # [batch_size, num_positions, input_dim, hidden_dim]

#         output = torch.einsum('bij,bijk->bik', x, weight)  # [batch_size, num_positions, hidden_dim]
#         return output


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class IO_FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2( self.act(self.fc1(x)) )


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def register_block_size(self, block_size):
        self.block_size = block_size

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        if input_pos.shape[0] == k_val.shape[2]:
            # predict_block = False
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val
        else:
            # predict_block = True
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, input_pos*self.block_size:(input_pos+1)*self.block_size] = k_val
            v_out[:, :, input_pos*self.block_size:(input_pos+1)*self.block_size] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None, 
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        # xq = apply_rotary_emb(xq, freqs_cis)
        # xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self, x: torch.Tensor, mask=None, input_pos=None):
        h = x + self.attention(x=self.attention_norm(x), mask=mask, input_pos=input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out