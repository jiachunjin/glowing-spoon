import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader


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