import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_autoencoder import SelfAttnBlock, create_decoder_attn_mask

class Autoencoder_1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder_1D(config.encoder)
        self.decoder = Decoder_1D(config.decoder)
    
    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        """
        x_BLD: vae features, L should be 256 for 16x16
        return: reconstructed vae features for all the levels
        """
        latents = self.encoder(x_BLD)
        recons = self.decoder(latents)
        return recons


class Encoder_1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 0. hyper parameters
        self.config = config
        self.embed_dim = config.embed_dim
        self.vae_dim = config.vae_dim
        self.output_dim = config.output_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.num_latents = config.num_latents
        self.num_patches = 256
        self.patch_size = 1
        scale = self.embed_dim ** -0.5

        # 1. token embedding for the vae features
        self.patch_embed = nn.Linear(self.vae_dim, self.embed_dim)

        # 2. positional embeddings
        # TODO: change to 2D ROPE?
        self.pos_embed = nn.Parameter(scale * torch.randn(self.num_patches, self.embed_dim))
        # self.latents_pos_embed = nn.Parameter(scale * torch.randn(self.num_latents, self.embed_dim))

        # 3. latent tokens
        self.latents = nn.Parameter(scale * torch.randn(self.num_latents, self.embed_dim))

        # 4. transformer
        self.norm_pre = nn.LayerNorm(self.embed_dim)
        self.transformer = nn.ModuleList([
            SelfAttnBlock(self.embed_dim, self.num_heads, mlp_ratio=4.0)
            for _ in range(self.num_layers)
        ])

        # 5. output latents
        self.norm_post = nn.LayerNorm(self.embed_dim)
        self.output = nn.Linear(self.embed_dim, self.output_dim)
    
    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        """
        x_BLD: (B, L, D), VAE feature map, L=256, D=16
        return: (B, K, D), latent tokens
        """
        B, L, D = x_BLD.shape
        dtype = x_BLD.dtype
        x_BLD = self.patch_embed(x_BLD)
        x_BLD = x_BLD + self.pos_embed.to(dtype)

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        # latents = latents + self.latents_pos_embed.to(dtype)

        x = torch.cat([x_BLD, latents], dim=1) # (B, L+K, D)

        x = self.norm_pre(x)
        for _, block in enumerate(self.transformer):
            x = block(x)

        latents = x[:, L:]
        latents = self.norm_post(latents)
        latents = self.output(latents)

        return latents


class Decoder_1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 0. hyper parameters
        self.config = config
        self.binary_mode = config.binary_mode
        self.embed_dim = config.embed_dim
        self.vae_dim = config.vae_dim
        self.input_dim = config.input_dim # the dim of the latents
        self.output_dim = config.input_dim # the dim of the latents
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.num_latents = config.num_latents
        self.recon_levels = config.recon_levels
        self.recon_length = sum(pn ** 2 for pn in self.recon_levels)
        attn_size = self.recon_length + self.num_latents
        scale = self.embed_dim ** -0.5
        print(f'Decoder binary_mode: {self.binary_mode}')

        # 1. input latent logits
        self.input = nn.Linear(self.input_dim, self.embed_dim)

        # 2. positional embeddings
        n = self.recon_levels[-1]
        self.pos_embed_full = nn.Parameter(scale * torch.randn(1, n ** 2, self.embed_dim))
        self.latents_pos_embed = nn.Parameter(scale * torch.randn(self.num_latents, self.embed_dim))

        # 2. mask tokens
        # one token for one level
        self.mask_tokens = nn.Parameter(scale * torch.randn(len(self.recon_levels), self.embed_dim))
        # register buffer for the attn mask
        attn_pair = []
        cur_left = 0
        for l in self.recon_levels:
            neighbours = []
            for i in range(l * l):
                neighbours.append(cur_left + i)
            for i in range(l * l):
                neighbours.append(self.recon_length+i)
            cur_left += l * l
            attn_pair.append(neighbours)
            if l == self.recon_levels[-1]:
                neighbours += list(range(self.recon_length+i, self.recon_length + self.num_latents))
        attn_mask = create_decoder_attn_mask(attn_pair, attn_size, self.recon_length).reshape(1, 1, attn_size, attn_size)
        self.register_buffer('attn_mask', attn_mask.contiguous())

        # 3. transformer
        self.norm_pre = nn.LayerNorm(self.embed_dim)
        self.transformer = nn.ModuleList([
            SelfAttnBlock(self.embed_dim, self.num_heads, mlp_ratio=4.0)
            for _ in range(self.num_layers)
        ])

        # 4. output latents
        self.norm_post = nn.LayerNorm(self.embed_dim)
        self.output = nn.Linear(self.embed_dim, self.vae_dim)

    def forward(self, latents_BKd: torch.Tensor) -> torch.Tensor:
        """
        latents_BKd: (B, K, d), output of the encoder, should be logits for Bernoulli
        return: (B, recon_length, vae_dim), reconstructed VAE feature map
        """
        B, K, d = latents_BKd.shape
        dtype = latents_BKd.dtype
        if self.binary_mode:
            # get bits from logits and embed
            p = F.sigmoid(latents_BKd)
            p_ = torch.bernoulli(p).to(p.dtype) # (B, K, d)
            latents_bin = p + (p_ - p).detach()
            x_BKD = self.input(latents_bin)
        else:
            # use continuous latent for experimental purpose
            x_BKD = self.input(latents_BKd)

        x_BKD = x_BKD + self.latents_pos_embed.to(dtype)
        
        mask_tokens = torch.cat([
            self.mask_tokens[i].repeat(self.recon_levels[i] ** 2, 1) for i in range(len(self.recon_levels))
        ], dim=0).unsqueeze(0).expand(B, -1, -1)

        # get interpolated position embeddings for each level
        n = self.recon_levels[-1]
        dtype = self.pos_embed_full.dtype
        pos_embed_full = self.pos_embed_full.float()  # interpolate needs float32
        pos_embed_full_square = pos_embed_full.reshape(1, n, n, self.embed_dim).permute(0, 3, 1, 2)
        pos_embed_all = []
        for level in self.recon_levels[:-1]:
            pos_embed = F.interpolate(pos_embed_full_square, size=(level, level), mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed_all.append(pos_embed)
        pos_embed_all.append(pos_embed_full)
        pos_embed_all = torch.cat(pos_embed_all, dim=1)
        mask_tokens = mask_tokens + pos_embed_all.to(mask_tokens.device, dtype)

        x = torch.cat([mask_tokens, x_BKD], dim=1) # (B, recon_length+K, D)
        x = self.norm_pre(x)
        for _, block in enumerate(self.transformer):
            x = block(x, attn_bias=self.attn_mask)
        
        recons = x[:, :self.recon_length]
        recons = self.norm_post(recons)
        recons = self.output(recons)
        
        return recons
        

if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load('configs/mar_vae.yaml')
    # encoder = Encoder_1D(config.encoder)
    # decoder = Decoder_1D(config.decoder)

    # num_p_encoder = sum(p.numel() for p in encoder.parameters())
    # num_p_decoder = sum(p.numel() for p in decoder.parameters())

    # print(f'encoder: {num_p_encoder}, decoder: {num_p_decoder}')

    autoencoder = Autoencoder_1D(config)
    num_p = sum(p.numel() for p in autoencoder.parameters())
    print(num_p)

    x = torch.randn(1, 256, 16)
    # latents = encoder(x)
    recons = autoencoder(x)
    print(recons.shape)



