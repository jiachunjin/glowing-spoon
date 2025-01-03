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
        wds.WebDataset(config.path, resampled=True, shardshuffle=False, nodesplitter=None)
        .shuffle(2048)
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
