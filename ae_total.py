import re
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mar_vae import Encoder, Decoder, DiagonalGaussianDistribution
from autoencoder import Encoder_1D, Decoder_1D_Matryoshka
from basic_autoencoder import SelfAttnBlock, create_decoder_attn_mask
from utils import get_latents_mask


class AE_total(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # mar encoder
        self.encoder = Encoder(ch_mult=(1, 1, 2, 2, 4), z_channels=16)
        self.decoder = Decoder(ch_mult=(1, 1, 2, 2, 4), z_channels=16)
        self.quant_conv = torch.nn.Conv2d(2 * 16, 2 * 16, 1)
        self.post_quant_conv = torch.nn.Conv2d(16, 16, 1)
        # 1d autoencoder
        self.encoder_1d = Encoder_1D(config.encoder_1d)
        self.decoder_1d = Decoder_1D_Matryoshka(config.decoder_1d)

        self._init_from_ckpt(config)

        # temporally switch off the grad for mar encoder
        self.encoder.requires_grad_(False)
    
    def _init_from_ckpt(self, config):
        if config.outer_ckpt is not None:
            sd = torch.load(config.outer_ckpt, map_location="cpu", weights_only=True)["model"]
            self.load_state_dict(sd, strict=False)

        if config.inner_ckpt is not None:
            sd = torch.load(config.inner_ckpt, map_location="cpu", weights_only=True)
            sd = {k.replace('encoder.', 'encoder_1d.'): v for k, v in sd.items()}
            sd = {k.replace('decoder.', 'decoder_1d.'): v for k, v in sd.items()}
            if config.skipped_keys:
                sd = {k: v for k, v in sd.items() if k not in config.skipped_keys}

            self.load_state_dict(sd, strict=False)

    def get_feature_1d(self, x_BCHW):
        h = self.encoder(x_BCHW)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        feature_map = posterior.mode()

        return rearrange(feature_map, 'b c h w -> b (h w) c')
    
    def forward(self, x_BCHW, num_activated_latent=None):
        feature_1d = self.get_feature_1d(x_BCHW)
        latents = self.encoder_1d(feature_1d)
        if self.training:
            feature_1d_recon_full = self.decoder_1d(latents, num_activated_latent=self.config.num_latents)
            feature_1d_recon_matryoshka = self.decoder_1d(latents, num_activated_latent=None)
            z_full = rearrange(feature_1d_recon_full, 'b (h w) c -> b c h w', h=16)
            z_full = self.post_quant_conv(z_full)
            z_matryoshka = rearrange(feature_1d_recon_matryoshka, 'b (h w) c -> b c h w', h=16)
            z_matryoshka = self.post_quant_conv(z_matryoshka)
            recon_full = self.decoder(z_full)
            recon_matryoshka = self.decoder(z_matryoshka)

            recon = (recon_full, recon_matryoshka)
        else:
            feature_1d_recon = self.decoder_1d(latents, num_activated_latent=num_activated_latent)
            z = rearrange(feature_1d_recon, 'b (h w) c -> b c h w', h=16)
            z = self.post_quant_conv(z)
            recon = self.decoder(z)

        return recon

    @torch.no_grad()
    def get_probs_and_bits(self, x_BCHW, latent_mask=None):
        feature_1d = self.get_feature_1d(x_BCHW)
        latents = self.encoder_1d(feature_1d)
        probs = F.sigmoid(latents)
        bits = torch.bernoulli(probs)
        bits = bits * 2 - 1 # from {0, 1} to {-1, 1}
        if latent_mask is not None:
            bits = bits * latent_mask

        return probs, bits
    
    @torch.no_grad()
    def decode_bits(self, bits, num_activated_latent=None):
        feature_1d_recon_full = self.decoder_1d(bits, num_activated_latent=num_activated_latent, decode_bits=True)
        z_full = rearrange(feature_1d_recon_full, 'b (h w) c -> b c h w', h=16)
        z_full = self.post_quant_conv(z_full)
        recons = self.decoder(z_full)
        
        return recons


if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import ToTensor
    from omegaconf import OmegaConf

    config = OmegaConf.load('configs/ae_total.yaml')
    ae = AE_total(config=config)

    num_p = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    print(f'Number of learnable parameters: {num_p}')

    # image = Image.open('assets/bear.png')
    # image = ToTensor()(image).unsqueeze(0)
    # image = image * 2 - 1
    # recon = ae(image)

    # recon = (recon + 1) / 2