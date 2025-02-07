import os
import argparse
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from einops import rearrange

from utils import get_dataloader, flatten_dict, bernoulli_entropy, get_latents_mask


def get_models(config):
    from gpt import Transformer_bin
    from autoencoder import Autoencoder_1D
    from mar_vae import AutoencoderKL
    from ae_total import AE_total

    if config.total:
        autoencoder = AE_total(config=config.autoencoder)
        ckpt = torch.load(config.autoencoder.pretrained_ckpt_path, map_location='cpu', weights_only=True)
        autoencoder.load_state_dict(ckpt)
        autoencoder.requires_grad_(False)
        autoencoder.eval()
        gpt = Transformer_bin(config.gpt)

        return autoencoder, gpt
    else:
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

def main(config_path):
    config = OmegaConf.load(config_path)
    accelerator, output_dir = get_accelerator(config.train)
    if config.total:
        autoencoder, gpt = get_models(config)
    else:
        autoencoder, vae, gpt = get_models(config)
    dataloader = get_dataloader(config.data)
    global_step = config.train.global_step if config.train.global_step is not None else 0

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location='cpu', weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = gpt.load_state_dict(ckpt, strict=False)
        # print('missing: ', m)
        print('unexpected: ', u)
        if gpt.config.independent_projection:
            gpt.input_proj.load_from_linear(ckpt['tok_eb.weight'], ckpt['tok_eb.bias'])
            gpt.output_proj.load_from_linear(ckpt['output.weight'], ckpt['output.bias'])
            print('initialized independent projection')
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
    if not config.total:
        vae = vae.to(accelerator.device)
    autoencoder = autoencoder.to(accelerator.device)

    if accelerator.is_main_process:
        if config.train.report_to == 'wandb':
            accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))
        else:
            accelerator.init_trackers(config.train.wandb_proj)

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
        latent_mask = get_latents_mask(
            num_latents = config.autoencoder.num_latents,
            input_dim   = config.autoencoder.binary_dim,
            schedule    = config.autoencoder.decoder_1d.latents_mask_schedule,
        )
        latent_mask = latent_mask.unsqueeze(0).to(accelerator.device)

    while not training_done:
        for x, y in dataloader:
            gpt.train()
            with accelerator.accumulate([gpt]):
                with torch.no_grad():
                    if config.total:
                        probs, bits = autoencoder.get_probs_and_bits(x, latent_mask=latent_mask)
                    else:
                        features_Bld = vae.get_feature(x)
                        probs, bits = autoencoder.get_probs_and_bits(features_Bld, latent_mask=latent_mask)
                    probs = probs[:, :num_pred_bits, :]
                    bits = bits[:, :num_pred_bits, :]
                    # random flip the bits
                    flip_mask = torch.rand_like(bits, dtype=torch.float, device=bits.device) > config.train.flip_prob
                    flip_mask = flip_mask.float() * 2 - 1 # {0, 1} -> {-1, 1}
                    bits *= flip_mask
                    entropy = bernoulli_entropy(probs)

                cond_idx = y.long()

                _, loss = gpt(binary_vec=bits, cond_idx=cond_idx, targets=probs)
                if 'latents_mask_schedule' in config.autoencoder.decoder_1d:
                    loss *= latent_mask[:, :num_pred_bits]
                    entropy *= latent_mask[:, :num_pred_bits]
                kl_divergence = (loss - entropy)
                kl_divergence_pre_128 = (loss - entropy)[:, :128]

                optimizer.zero_grad()
                accelerator.backward(loss.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                optimizer.step()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                loss = accelerator.gather(loss.detach()).mean().item()
                kl_divergence = accelerator.gather(kl_divergence.detach()).mean().item()
                kl_divergence_pre_128 = accelerator.gather(kl_divergence_pre_128.detach()).mean().item()

                logs = dict(
                    loss          = loss,
                    kl_divergence = kl_divergence,
                    kl_divergence_pre_128 = kl_divergence_pre_128,
                )
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                gpt.eval()
                state_dict = accelerator.unwrap_model(gpt).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"GPT-{config.train.exp_name}-{global_step // 1000}k"))
            accelerator.wait_for_everyone()

            if global_step > 0 and global_step % config.train.val_every == 0:
                from torchvision import transforms
                from generate import generate_blockwise
                cfg_scale = 1

                gpt.eval()
                rank = accelerator.state.local_process_index
                world_size = accelerator.state.num_processes

                labels = torch.arange(rank, 1000, world_size).to(accelerator.device)

                inverse_transform = transforms.Compose([
                    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
                    transforms.Lambda(lambda x: x.clamp(0, 1)),
                    transforms.ToPILImage()
                ])

                with torch.no_grad():
                    for lable in tqdm(labels):
                        cond = torch.tensor([lable]*50).to(accelerator.device)
                        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                            out = generate_blockwise(gpt.module, cond, 1024, cfg_scale, latent_mask, accelerator.device, verbose=False)
                        with torch.no_grad(), torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                            recon_full = autoencoder.decode_bits(out, num_activated_latent=None)
                            for id, rec in enumerate(recon_full):
                                rec = inverse_transform(rec)
                                rec.save(f'eval_imgs/gen/{rank}_{lable}_{id}.png')                
                print('done')
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    import subprocess
                    command = [
                        "python", "-m", "pytorch_fid",
                        "/root/codebase/gpt/glowing-spoon/eval_imgs/ori",
                        "/root/codebase/gpt/glowing-spoon/eval_imgs/gen",
                        "--device", "cuda:7"
                    ]

                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    for line in process.stdout:
                        print(line)
                    fid = float(line.split()[-1])
                    # 等待进程结束
                    process.wait()
                    fid_log = {'FID': fid}
                    accelerator.log(fid_log, step=global_step)
                    print(f'FID at {global_step}: {fid}')
            accelerator.wait_for_everyone()

            if global_step >= config.train.num_iters:
                training_done = True
                break
    accelerator.end_training()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ae_total.yaml')
    args = parser.parse_args()
    main(args.config)