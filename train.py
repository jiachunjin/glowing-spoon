import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from utils import get_dataloader, flatten_dict

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

    while not training_done:
        for x, y in dataloader:
            autoencoder.train()
            with accelerator.accumulate([autoencoder]):
                with torch.no_grad():
                    features_Bld, targets_BLd = vae.get_multi_level_features(x, config.autoencoder.decoder.recon_levels)
                recons = autoencoder(features_Bld)
                loss = F.mse_loss(recons, targets_BLd, reduction='mean')

                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                
                accelerator.backward(loss)
                optimizer.step()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                loss = accelerator.gather(loss.detach()).mean().item()
                logs = {'loss': loss}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                autoencoder.eval()
                state_dict = accelerator.unwrap_model(autoencoder).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"AE-{config.train.exp_name}-{global_step // 1000}k"))

            if global_step >= config.train.num_iters:
                training_done = True
                break


if __name__ == '__main__':
    main()