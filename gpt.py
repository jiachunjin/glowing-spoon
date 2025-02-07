import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from einops import rearrange, repeat 

from basic_gpt import LabelEmbedder, TransformerBlock, RMSNorm, KVCache, Independent_Projection, IO_FFN


class Transformer_bin(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 0. hyperparameters
        self.config = config
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.num_classes = config.num_classes
        self.cls_token_num = config.cls_token_num
        self.independent_projection = config.independent_projection
        if config.block_size is None:
            self.block_prediction = False
            self.seq_len = config.seq_len + 1
        else:
            self.block_prediction = True
            self.block_size = config.block_size
            self.seq_len = config.seq_len + self.block_size
        scale = self.dim ** -0.5

        self.cls_embedding = LabelEmbedder(self.num_classes, self.dim, config.class_dropout_prob)
        self.pos_embedding = nn.Parameter(scale * torch.randn(self.seq_len, self.dim)) # TODO
        # self.pos_embedding = nn.Parameter(scale * torch.randn(681, self.dim)) # TODO
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        if config.independent_projection:
            self.input_proj = Independent_Projection(config.seq_len, config.input_dim, config.dim)
            self.output_proj = Independent_Projection(config.seq_len, config.dim, config.input_dim)
        else:
            # self.tok_eb = IO_FFN(config.input_dim, config.dim, config.dim)
            # self.output = IO_FFN(config.dim, config.dim, config.input_dim)
            self.tok_eb = nn.Linear(config.input_dim, config.dim)
            self.output = nn.Linear(config.dim, config.input_dim) # TODO

        self.layers = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.layers.append(TransformerBlock(config))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        if self.block_prediction:
            mask_size = self.seq_len
            block_size = self.block_size
            mask = torch.full((mask_size, mask_size), -torch.inf)
            for i in range(0, mask_size, block_size):
                for j in range(0, i + block_size, block_size):
                    if i >= j:
                        mask[i:i+block_size, j:j+block_size] = 0
            self.register_buffer('mask', mask.reshape(1, 1, mask_size, mask_size))

    def forward(
        self, 
        binary_vec,
        cond_idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        if binary_vec is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            if self.block_prediction:
                # repeat self.block_size times to match the size of the first block
                cond_embeddings = cond_embeddings.repeat_interleave(self.block_size, dim=1)
                # build a block mask with block_size the same as the self.block_size
                mask = self.mask
                # patch_nums = [4] * (self.seq_len // self.block_size)
                # L = sum(pn ** 2 for pn in patch_nums)
                # d = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(patch_nums)]).view(1, L, 1)
                # dT = d.transpose(1, 2)
                # mask = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L).to(cond_embeddings.device)
            else:
                mask = None # this leads to the casual mask
            if self.independent_projection:
                token_embeddings = self.input_proj(binary_vec)
            else:
                token_embeddings = self.tok_eb(binary_vec)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            h += self.pos_embedding[:h.shape[1]]
        else:
            assert self.training==False
            if cond_idx is not None:
                # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=False)
                if self.block_prediction:
                    token_embeddings = token_embeddings.repeat_interleave(self.block_size, dim=1)
            else:
                # decode_n_tokens(kv cache) in inference
                if self.independent_projection:
                    token_embeddings = self.input_proj(binary_vec, input_pos)
                else:
                    token_embeddings = self.tok_eb(binary_vec)
            if not self.block_prediction:
                bs = token_embeddings.shape[0]
                mask = self.causal_mask[:bs, None, input_pos]
                h = self.tok_dropout(token_embeddings)
                h += self.pos_embedding[input_pos]
            else:
                level_idx = input_pos
                bs = token_embeddings.shape[0]
                mask = self.mask[:, :, level_idx * self.block_size:(level_idx + 1) * self.block_size]
                h = self.tok_dropout(token_embeddings)
                h += self.pos_embedding[level_idx * self.block_size:(level_idx + 1) * self.block_size]

        for layer in self.layers:
            seq_len = h.shape[1]
            # h = layer(h, mask=mask[:, :, :seq_len, :seq_len], input_pos=input_pos)
            h = layer(h, mask=mask[:, :, :, :], input_pos=input_pos)

        h = self.norm(h)
        if input_pos is None:
            if self.block_prediction:
                if self.independent_projection:
                    logits = self.output_proj(h[:, :-self.block_size, :]).float()
                else:
                    logits = self.output(h[:, :-self.block_size, :]).float()
            else:
                logits = self.output_proj(h[:, :-1, :]).float()
        else:
            assert self.training==False
            if self.independent_projection:
                logits = self.output_proj(h, input_pos).float()
            else:
                logits = self.output(h).float()
        
        loss = None
        if valid is not None:
            raise NotImplementedError("Not implemented yet")
        elif targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        return logits, loss
    
    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)
        if not self.block_prediction:
            causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
            self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
            # print(f'block_prediction: {self.block_prediction}, setup_cache done')
        else:
            for b in self.layers:
                b.attention.kv_cache.register_block_size(self.block_size)
            # print(f'block_prediction: {self.block_prediction}, setup_cache done')


if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load('configs/gpt.yaml')
    gpt = Transformer_bin(config.gpt)

    binary_vec = torch.randint(0, 2, (2, 680, 16), dtype=torch.float32)
    cond_idx = torch.randint(0, 1000, (2,)).long()

    targets = torch.rand((2, 680, 16), dtype=torch.float32)
    logits, loss = gpt(binary_vec, cond_idx, targets=targets)
    print(logits.shape, loss.shape)