import math
import numpy as np
import torch
import webdataset as wds
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from einops import rearrange


def get_dataloader(config):
    llamagen_transform = transforms.Compose([
        transforms.Resize(256, max_size=None),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = (
        wds.WebDataset(config.path, resampled=True, shardshuffle=True, nodesplitter=None)
        .shuffle(128000) # 1/10 of the training set
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(llamagen_transform, None)
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=8, pin_memory=True)
    
    return dataloader

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # 如果值是字典，则递归调用 flatten_dict
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_loss_per_level(loss, levels):
    loss = loss.mean(dim=[0,2])
    loss_per_level = []
    cur_idx = 0
    for i, level in enumerate(levels):
        ll = loss[cur_idx:cur_idx+level**2].mean().detach().item()
        loss_per_level.append(ll)
        cur_idx += level**2
    
    return loss_per_level

def get_loss_weighting(levels=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16]):
    L = sum(level ** 2 for level in levels)
    weight = torch.ones((L,))
    curr_idx = 0
    for i, level in enumerate(levels):
        weight[curr_idx:curr_idx+level**2] = level ** 2
        curr_idx += level**2
    return weight / levels[-1]**2

def get_residual_summation(recons, levels):
    B = recons.shape[0]
    cur_idx = 0

    # first way of residual summation
    # prev_h = None
    # for level in levels:
    #     cur_h = rearrange(recons[:, cur_idx:cur_idx+level**2], 'b (h w) c -> b c h w', h=level)
    #     if prev_h is not None:
    #         cur_h += F.interpolate(prev_h, size=(level, level), mode='area')
    #     cur_idx += level**2
    #     prev_h = cur_h
    
    # return cur_h

    # another way of residual summation
    f = torch.zeros((B, 16, 16, 16)).to(recons.device)
    for level in levels:
        cur_h = rearrange(recons[:, cur_idx:cur_idx+level**2], 'b (h w) c -> b c h w', h=level)
        up_h = F.interpolate(cur_h, size=(16, 16), mode='area')
        f += up_h
        
        cur_idx += level**2
        
    return f

def bernoulli_entropy(p):
    return -p * torch.log(torch.clip(p, 1e-10, 1)) - (1 - p) * torch.log(torch.clip(1 - p, 1e-10, 1))

def get_latents_mask(num_latents, input_dim, schedule):
    mask = torch.zeros(num_latents, input_dim)
    if schedule == 'linear_16':
        # 1, 2, 3, 4, ..., 15, 16 (16x32 = 512)
        for i in range(16):
            start = i * 32
            end = (i + 1) * 32
            mask[start:end, :i + 1] = 1
    elif schedule == 'linear_24':
        # [ 1  2  3  4  4  5  6  7  7  8  9 10 10 11 12 13 13 14 15 16 16 17 18 19 19 20 21 22 22 23 24 24]
        # 6640
        num_blocks = 32
        block_size = 16
        max_bits = 24
        min_bits = 2
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, 24, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'step_linear_32':
        # [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32] 8448
        num_blocks = 32
        block_size = 16
        max_bits = 32
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, 32, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'linear_64':
        num_blocks = 64
        block_size = 8
        max_bits = 64
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'linear_768_32':
        num_blocks = 64
        block_size = 12
        max_bits = 32
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'linear_1024_32':
        num_blocks = 64
        block_size = 16
        max_bits = 32
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == '4_to_32':
        num_blocks = 64
        block_size = 8
        max_bits = 32
        min_bits = 4
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(min_bits, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == '2_to_32':
        num_blocks = 64
        block_size = 8
        max_bits = 32
        min_bits = 2
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(min_bits, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == '1_to_32_768':
        num_blocks = 48
        block_size = 16
        max_bits = 32
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(min_bits, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == '1_to_16_512':
        num_blocks = 32
        block_size = 16
        max_bits = 16
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(min_bits, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'linear_1024_36':
        num_blocks = 64
        block_size = 16
        max_bits = 36
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
        # print(num_activated_bits)
    elif schedule == 'linear_1536_16':
        num_blocks = 64
        block_size = 24
        max_bits = 16
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'linear_2048_16':
        num_blocks = 64
        block_size = 32
        max_bits = 16
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'linear_1024_16':
        num_blocks = 64
        block_size = 16
        max_bits = 16
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'linear_1024_24':
        num_blocks = 64
        block_size = 16
        max_bits = 24
        min_bits = 1
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.ceil(np.linspace(1, max_bits, num_blocks)).astype(int)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'exp_32':
        # [2 3 4 6 8 12 18 26 32 32 32 32 32 32 32 32] tensor(10720.) 512x16 = 8192, 16x16x64 = 16384
        l = 16
        num_l = 32
        assert num_latents == l * num_l
        assert input_dim == 32
        num_activated_bits = np.minimum(np.array([math.ceil(1.5**(i)) for i in range(1, l+1)]), 32)
        for i in range(l):
            start = i * num_l
            end = (i + 1) * num_l
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'exp_48':
        # [ 2  2  3  4  5  6  7 10 13 17 22 28 37 48 48 48] tensor(9600.)
        num_blocks = 16
        block_size = 32
        max_bits = 48
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.minimum(np.array([math.ceil(1.32**(i)) for i in range(1, num_blocks+1)]), max_bits)
        print(num_activated_bits)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    elif schedule == 'flat_exp_32':
        # [ 4  4  4  4  5  6  7  8  9 11 13 16 19 23 27 32] tensor(6144.)
        num_blocks = 16
        block_size = 32
        max_bits = 32
        min_bits = 4
        mask = torch.zeros(num_blocks*block_size, max_bits)
        num_activated_bits = np.minimum(np.array([math.ceil(1.2**(i)) for i in range(4, num_blocks+4)]), max_bits)
        num_activated_bits = np.maximum(num_activated_bits, min_bits)
        print(num_activated_bits)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, :num_activated_bits[i]] = 1
    else:
        raise ValueError(f'Unknown schedule: {schedule}')
    
    return mask

import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.state_dict().items()}
        
    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:  # Only update floating-point parameters
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()
    
    def apply_shadow(self):
        """
        Replace the model's parameters with the EMA parameters.
        """
        self.model.load_state_dict(self.shadow)

    def save_shadow(self, path: str):
        torch.save(self.shadow, path)