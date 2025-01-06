import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from einops import rearrange

from utils import get_dataloader, flatten_dict, bernoulli_entropy


def get_models(config):
    from gpt import Transformer_bin
    from autoencoder import Autoencoder_1D
    from mar_vae import AutoencoderKL

    autoencoder = Autoencoder_1D(config.autoencoder)
    ckpt = torch.load(config.autoencoder.pretrained_ckpt_path, map_location='cpu')
    autoencoder.load_state_dict(ckpt)
    autoencoder.requires_grad_(False)
    autoencoder.eval()
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path='pretrained_models/vae/kl16.ckpt')
    vae.requires_grad_(False)
    vae.eval()

    gpt = Transformer_bin(config.gpt)

    return autoencoder, vae, gpt

def get_accelerator(config):
    output_dir = os.path.join('experiment', config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        project_config              = project_config,
        mixed_precision             = config.mixed_precision,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        log_with                    = None if config.report_to == 'no' else config.report_to,
    )

    return accelerator, output_dir

def main():
    config = OmegaConf.load('configs/gpt.yaml')
    accelerator, output_dir = get_accelerator(config.train)
    autoencoder, vae, gpt = get_models(config)
    dataloader = get_dataloader(config.data)
    global_step = config.train.global_step if config.train.global_step is not None else 0

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location='cpu')
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = gpt.load_state_dict(ckpt, strict=False)
        print('missing: ', m)
        print('unexpected: ', u)
        if accelerator.is_main_process:
            print(f'GPT ckpt loaded from {config.train.resume_path}')
    
    params_to_learn = list(gpt.parameters())
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = 1e-4,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )
    if accelerator.is_main_process:
        print('Number of learnable parameters: ', sum(p.numel() for p in params_to_learn if p.requires_grad))

    gpt, dataloader, optimizer = accelerator.prepare(gpt, dataloader, optimizer)
    vae = vae.to(accelerator.device)
    autoencoder = autoencoder.to(accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))

    training_done = False
    progress_bar = tqdm(
        total   = config.train.num_iters,
        initial = global_step,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )
    if config.train.num_pred_bits is not None:
        num_pred_bits = config.train.num_pred_bits
    else:
        num_pred_bits = config.autoencoder.num_latents
    if accelerator.is_main_process:
        print(f'Number of predicted tokens: {num_pred_bits}')

    if config.autoencoder.matryoshka:
        latent_mask = torch.zeros(config.autoencoder.decoder_matryoshka.num_latents, config.autoencoder.decoder_matryoshka.input_dim).to(accelerator.device)
        assert config.autoencoder.decoder_matryoshka.num_latents == 512, 'only support 512 latents for now'
        assert config.autoencoder.decoder_matryoshka.input_dim == 16, 'only support 16 dim latents for now'
        for i in range(16):
            start = i * 32
            end = (i + 1) * 32
            latent_mask[start:end, :i + 1] = 1
        latent_mask = latent_mask.unsqueeze(0)

    while not training_done:
        for x, y in dataloader:
            gpt.train()
            with accelerator.accumulate([gpt]):
                with torch.no_grad():
                    features_Bld = vae.get_feature(x)
                    probs, bits = autoencoder.get_probs_and_bits(features_Bld, latent_mask=latent_mask)
                    probs = probs[:, :config.train.num_pred_bits, :]
                    bits = bits[:, :config.train.num_pred_bits, :]
                    entropy = bernoulli_entropy(probs)

                cond_idx = y.long()

                _, loss = gpt(binary_vec=bits, cond_idx=cond_idx, targets=probs)
                if 'latents_mask_schedule' in config.autoencoder.decoder_matryoshka:
                    loss *= latent_mask
                    entropy *= latent_mask
                kl_divergence = (loss - entropy)

                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                
                accelerator.backward(loss.mean())
                optimizer.step()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                loss = accelerator.gather(loss.detach()).mean().item()
                kl_divergence = accelerator.gather(kl_divergence.detach()).mean().item()

                logs = dict(
                    loss          = loss,
                    kl_divergence = kl_divergence,
                )
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                gpt.eval()
                state_dict = accelerator.unwrap_model(gpt).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"GPT-{config.train.exp_name}-{global_step // 1000}k"))
            accelerator.wait_for_everyone()

            if global_step >= config.train.num_iters:
                training_done = True
                break
    accelerator.end_training()

    


if __name__ == '__main__':
    main()