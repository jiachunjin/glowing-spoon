import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from mar_vae import AutoencoderKL

##### load pretrained VAE #####
vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path='pretrained_models/vae/kl16.ckpt').eval()

##### reconstruct an image with VAE #####
def reconstruct_img(vae):
    img = Image.open('assets/bear.png')
    img = ToTensor()(img).unsqueeze(0)
    img = img * 2 - 1

    posterior = vae.encode(img)
    h = posterior.mode()
    print(h.shape)
    rec = vae.decode(h)
    rec = torch.clamp((rec + 1) / 2, 0, 1)

    ToPILImage()(rec[0]).save('assets/bear_rec.png')

def test_multi_level_interpolation(vae):
    img = Image.open('assets/bear.png')
    img = ToTensor()(img).unsqueeze(0).repeat(2, 1, 1, 1)
    img = img * 2 - 1

    levels = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    h, t = vae.get_multi_level_features(img, levels)
    print(h[0])

    return h, t

def reconstruct_validation():
    from mar_vae import AutoencoderKL
    from autoencoder import Autoencoder_1D

    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.datasets import ImageNet
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from omegaconf import OmegaConf
    from einops import rearrange

    device = torch.device('cuda:0')
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path='pretrained_models/vae/kl16.ckpt').eval()
    config = OmegaConf.load('configs/mar_vae.yaml')

    autoencoder = Autoencoder_1D(config.autoencoder)
    ckpt = torch.load('experiment/680x16_residual_bin/AE-distill_MAR_VAE-60k', map_location='cpu')
    autoencoder.load_state_dict(ckpt, strict=True)
    autoencoder.eval();
    vae = vae.to(device)
    autoencoder = autoencoder.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256, max_size=None),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    val_dataset = ImageNet(root='/data/Largedata/ImageNet', split='val', transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=1)

    idx_rec = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            features_Bld, _ = vae.get_multi_level_features(images, [1, 2, 3, 4, 5, 6, 8, 10, 13, 16], residual=True)
            recons = autoencoder(features_Bld)
            
            levels = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
            cur_idx = 0
            f = torch.zeros((images.shape[0], 16, 16, 16)).to(recons.device)
            for level in levels:
                cur_h = rearrange(recons[:, cur_idx:cur_idx+level**2], 'b (h w) c -> b c h w', h=level)
                up_h = F.interpolate(cur_h, size=(16, 16), mode='area')
                f += up_h
                cur_idx += level**2
            recs = vae.decode(f)
            recs = torch.clamp((recs + 1) / 2, 0, 1)

            for rec in recs:
                rec_img = ToPILImage()(rec)
                rec_img.save(f'assets/recons/{idx_rec}.png')
                idx_rec += 1

if __name__ == '__main__':
    reconstruct_validation()