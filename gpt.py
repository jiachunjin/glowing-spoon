import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from einops import rearrange, repeat 

from basic_gpt import LabelEmbedder, TransformerBlock, RMSNorm, KVCache


class Transformer_bin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.num_classes = config.num_classes
        self.cls_token_num = config.cls_token_num
        self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)

        self.pos_embedding = nn.Parameter(torch.randn(680 + 1, config.dim))
        self.tok_eb = nn.Linear(config.input_dim, config.dim)

        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.input_dim, bias=False)

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
            token_embeddings = self.tok_eb(binary_vec)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            mask = None
            h = self.tok_dropout(token_embeddings)
            h += self.pos_embedding[:h.shape[1]]
        else:
            if cond_idx is not None:
                # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            else:
                # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_eb(binary_vec)
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            h += self.pos_embedding[input_pos]

        for layer in self.layers:
            h = layer(h, mask=mask, input_pos=input_pos)

        h = self.norm(h)
        if input_pos is None:
            logits = self.output(h[:, :-1, :]).float()
        else:
            assert self.training==False
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
        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        print('setup_cache done')


if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load('configs/gpt.yaml')
    gpt = Transformer_bin(config.gpt)

    binary_vec = torch.randint(0, 2, (2, 680, 16), dtype=torch.float32)
    cond_idx = torch.randint(0, 1000, (2,)).long()

    targets = torch.rand((2, 680, 16), dtype=torch.float32)
    logits, loss = gpt(binary_vec, cond_idx, targets=targets)
    print(logits.shape, loss.shape)