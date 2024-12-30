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
    vae.get_multi_level_features(img, levels)

if __name__ == '__main__':
    test_multi_level_interpolation(vae)