from numpy import block
import torch
import torch.nn.functional as F
from tqdm.auto import trange


@torch.no_grad()
def generate(gpt, cond, max_new_tokens, cfg_scale, device):
    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond) * gpt.num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond

    T = 1
    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        gpt.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=gpt.pos_embedding.dtype)

    seq = torch.empty((max_batch_size, T_new, gpt.config.input_dim), dtype=torch.float, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(gpt, cond_combined, input_pos, cfg_scale)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(gpt, next_token, input_pos, max_new_tokens-1, cfg_scale)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    return seq[:, T:]


def prefill(gpt, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, latent_mask=None):
    if cfg_scale > 1.0:
        logits, _ = gpt(None, cond_idx=cond_idx, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = gpt(None, cond_idx=cond_idx, input_pos=input_pos)
    
    bits = torch.bernoulli(F.sigmoid(logits))
    bits = bits * 2 - 1
    if latent_mask is not None:
        assert gpt.block_prediction
        block_size = gpt.block_size
        bits = bits * latent_mask[:, input_pos*block_size:(input_pos+1)*block_size]

    return bits


def decode_n_tokens(gpt, cur_token, input_pos, num_new_tokens, cfg_scale, latent_mask):
    new_tokens, new_probs = [], []
    for i in trange(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(gpt, cur_token, input_pos, cfg_scale, latent_mask)
            input_pos += 1
            new_tokens.append(next_token.clone())
            cur_token = next_token
    
    return new_tokens, new_probs


def decode_one_token(gpt, x, input_pos, cfg_scale, latent_mask=None):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = gpt(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = gpt(x, cond_idx=None, input_pos=input_pos)
   
    bits = torch.bernoulli(F.sigmoid(logits))
    # bits = F.sigmoid(logits) > 0.5
    bits = bits * 2 - 1

    if latent_mask is not None:
        assert gpt.block_prediction
        block_size = gpt.block_size
        bits = bits * latent_mask[:, input_pos*block_size:(input_pos+1)*block_size]

    return bits


@torch.no_grad()
def generate_blockwise(gpt, cond, max_new_tokens, cfg_scale, latent_mask, device):
    assert gpt.block_prediction
    block_size = gpt.block_size # 16
    num_blocks = gpt.seq_len // block_size
    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond) * gpt.num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond

    # T = 1
    # T_new = T + max_new_tokens
    max_seq_length = max_new_tokens + gpt.block_size
    max_batch_size = cond.shape[0]

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        gpt.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=gpt.pos_embedding.dtype)

    seq = torch.zeros((max_batch_size, max_new_tokens, gpt.config.input_dim), dtype=torch.float, device=device)

    level_idx = 0
    input_pos = torch.tensor([level_idx], device=device)
    next_token = prefill(gpt, cond_combined, input_pos, cfg_scale, latent_mask)
    seq[:, level_idx * block_size:(level_idx+1)*block_size] = next_token
    level_idx += 1

    input_pos = torch.tensor([level_idx], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(gpt, next_token, input_pos, num_blocks-2, cfg_scale, latent_mask)
    
    seq[:, (level_idx)*block_size:] = torch.cat(generated_tokens, dim=1)

    return seq