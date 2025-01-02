import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from einops import rearrange

from utils import get_dataloader, flatten_dict, get_loss_per_level, get_loss_weighting, get_residual_summation

def get_models(config):
    from autoencoder import Autoencoder_1D
    from mar_vae import AutoencoderKL
    autoencoder = Autoencoder_1D(config)
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path='pretrained_models/vae/kl16.ckpt')
    vae.requires_grad_(False)
    vae.eval()

    return autoencoder, vae

def get_accelerator(config):
    output_dir = os.path.join('experiment', config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=None if config.report_to == 'no' else config.report_to,
        mixed_precision=config.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    return accelerator, output_dir

def main():
    config = OmegaConf.load('configs/mar_vae.yaml')
    accelerator, output_dir = get_accelerator(config.train)
    autoencoder, vae = get_models(config.autoencoder)
    dataloader = get_dataloader(config.data)
    global_step = config.train.global_step if config.train.global_step is not None else 0

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location='cpu')
        # skipped_keys = ['encoder.latents', 'decoder.latents_pos_embed', 'decoder.attn_mask']
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = autoencoder.load_state_dict(ckpt, strict=False)
        print('missing: ', m)
        print('unexpected: ', u)
        if accelerator.is_main_process:
            print(f'AE ckpt loaded from {config.train.resume_path}')

    params_to_learn = list(autoencoder.parameters())
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = 1e-4,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )
    if accelerator.is_main_process:
        print('Number of learnable parameters: ', sum(p.numel() for p in params_to_learn if p.requires_grad))

    autoencoder, dataloader, optimizer = accelerator.prepare(autoencoder, dataloader, optimizer)
    vae = vae.to(accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))

    training_done = False
    progress_bar = tqdm(
        total=config.train.num_iters,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    lw = get_loss_weighting(config.autoencoder.decoder.recon_levels).to(accelerator.device) # (L,)

    while not training_done:
        for x, y in dataloader:
            autoencoder.train()
            with accelerator.accumulate([autoencoder]):
                with torch.no_grad():
                    features_Bld, targets_BLd = vae.get_multi_level_features(x, config.autoencoder.decoder.recon_levels, config.residual)
                recons = autoencoder(features_Bld)
                loss = F.mse_loss(recons, targets_BLd, reduction='none')
                loss_per_element = loss.mean(dim=[0,2]) # (B, L)
                if not config.residual:
                    weighted_loss = loss_per_element * lw
                else:
                    weighted_loss = loss_per_element
                    residual_summation = get_residual_summation(recons, config.autoencoder.decoder.recon_levels)
                    features = rearrange(features_Bld, 'b (h w) c -> b c h w', h=16)
                    complete_loss = F.mse_loss(residual_summation, features)
                    weighted_loss += complete_loss

                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                
                accelerator.backward(weighted_loss.mean())
                optimizer.step()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                loss = accelerator.gather(loss.detach())
                loss_per_level = get_loss_per_level(loss, config.autoencoder.decoder.recon_levels)
                loss = loss.mean().item()
                logs = {'loss': loss}
                for i, loss_pl in enumerate(loss_per_level):
                    logs[f'loss_{i}'] = loss_pl
                if config.residual:
                    complete_loss = accelerator.gather(complete_loss.detach()).mean().item()
                    logs['complete_loss'] = complete_loss
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            # if global_step > 0 and global_step % config.train.val_every == 0 and accelerator.is_main_process:
            #     from einops import rearrange
            #     from PIL import Image
            #     from torchvision.transforms import ToTensor, ToPILImage
            #     with torch.no_grad():
            #         rec = rearrange(recons[:, -256:, :], 'b (h w) c -> b c h w', h=16)
            #         rec = vae.decode(rec).detach().cpu()
            #         rec = torch.clamp((rec + 1) / 2, 0, 1)
            #         img = ToPILImage()(rec[0])
            #         img.save(os.path.join(output_dir, f"AE-{config.train.exp_name}-{global_step}.png"))

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                autoencoder.eval()
                state_dict = accelerator.unwrap_model(autoencoder).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"AE-{config.train.exp_name}-{global_step // 1000}k"))
            accelerator.wait_for_everyone()

            if global_step >= config.train.num_iters:
                training_done = True
                break
    accelerator.end_training()


if __name__ == '__main__':
    main()