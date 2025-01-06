import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from einops import rearrange, repeat 

from basic_gpt import LabelEmbedder, TransformerBlock, RMSNorm


class Transformer_bin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.block_size = config.block_size
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
    ):
        if binary_vec is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_eb(binary_vec)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
        else:
            raise NotImplementedError("Not implemented yet")
        
        h += self.pos_embedding[:h.shape[1]]

        for layer in self.layers:
            h = layer(h, mask=None)

        h = self.norm(h)
        logits = self.output(h[:, :-1, :]).float()

        
        loss = None
        if valid is not None:
            raise NotImplementedError("Not implemented yet")
        elif targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        return logits, loss


if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load('configs/gpt.yaml')
    gpt = Transformer_bin(config.gpt)

    binary_vec = torch.randint(0, 2, (2, 680, 16), dtype=torch.float32)
    cond_idx = torch.randint(0, 1000, (2,)).long()

    targets = torch.rand((2, 680, 16), dtype=torch.float32)
    logits, loss = gpt(binary_vec, cond_idx, targets=targets)
    print(logits.shape, loss.shape)