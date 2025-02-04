import os
import argparse
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from ae_total import AE_total
from hybrid_loss import Hybrid_Loss
from utils import EMA, get_dataloader, flatten_dict


def get_models(config):
    autoencoder = AE_total(config=config.autoencoder)
    hybrid_loss = Hybrid_Loss(disc_start=config.hybrid_loss.disc_start, disc_weight=config.hybrid_loss.disc_weight)

    return autoencoder, hybrid_loss

def get_accelerator(config):
    # output_dir = os.path.join('/data/experiment', config.output_dir)
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

def main(config_path):
    config = OmegaConf.load(config_path)
    accelerator, output_dir = get_accelerator(config.train)
    autoencoder, hybrid_loss = get_models(config)
    dataloader = get_dataloader(config.data)
    global_step = config.train.global_step if config.train.global_step is not None else 0

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location='cpu', weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = autoencoder.load_state_dict(ckpt, strict=False)
        print('missing: ', m)
        print('unexpected: ', u)
        # for n, p in autoencoder.named_parameters():
        #     if n in m:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False
        if accelerator.is_main_process:
            print(f'AE ckpt loaded from {config.train.resume_path}')

    if config.train.loss_resume_path is not None:
        ckpt = torch.load(config.train.loss_resume_path, map_location='cpu', weights_only=True)
        m, u = hybrid_loss.load_state_dict(ckpt, strict=False)
        print('missing: ', m)
        print('unexpected: ', u)
        if accelerator.is_main_process:
            print(f'Loss ckpt loaded from {config.train.loss_resume_path}')

    params_to_learn = list(autoencoder.parameters())
    disc_params = list(hybrid_loss.discriminator.parameters())

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = 1e-4,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )
    optimizer_disc = torch.optim.AdamW(
        disc_params,
        lr           = 1e-4 / config.hybrid_loss.disc_weight,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    if accelerator.is_main_process:
        print('Number of learnable parameters: ', sum(p.numel() for p in params_to_learn if p.requires_grad))
    
    autoencoder, hybrid_loss, dataloader, optimizer, optimizer_disc = accelerator.prepare(autoencoder, hybrid_loss, dataloader, optimizer, optimizer_disc)

    if accelerator.is_main_process:
        ema = EMA(autoencoder.module, decay=0.999)

    if accelerator.is_main_process:
        if config.train.report_to == 'wandb':
            accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))
        else:
            accelerator.init_trackers(config.train.wandb_proj)

    training_done = False
    progress_bar = tqdm(
        total=config.train.num_iters,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    epoch = 1
    while not training_done:
        print(f'epoch: {epoch}')
        for x, y in dataloader:
            autoencoder.train()
            hybrid_loss.train()
            with accelerator.accumulate([autoencoder, hybrid_loss]):
                recon_full, recon_matryoshka = autoencoder(x)
                # recon_full = autoencoder.module.forward_decoder_only(x)
                # --------------------- optimize autoencoder ---------------------
                loss_gen = hybrid_loss(
                    inputs          = x,
                    reconstructions = recon_full,
                    optimizer_idx   = 0,
                    global_step     = global_step+1,
                    last_layer      = autoencoder.module.decoder.last_layer
                )

                loss_matryoshka = F.mse_loss(recon_matryoshka, x, reduction='mean')
                # loss_matryoshka = 0

                optimizer.zero_grad()
                accelerator.backward(loss_gen + config.train.hp_matryoshka * loss_matryoshka)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                optimizer.step()
                if accelerator.is_main_process:
                    ema.update(autoencoder.module)
                # --------------------- optimize discriminator ---------------------
                loss_disc = hybrid_loss(
                    inputs          = x,
                    reconstructions = recon_full,
                    optimizer_idx   = 1,
                    global_step     = global_step+1,
                )
                optimizer_disc.zero_grad()
                accelerator.backward(loss_disc)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(disc_params, 1.0)
                optimizer_disc.step()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                loss_gen = accelerator.gather(loss_gen.detach()).mean().item()
                loss_disc = accelerator.gather(loss_disc.detach()).mean().item()
                # loss_matryoshka = accelerator.gather(loss_matryoshka.detach()).mean().item()

                logs = dict()
                logs['loss_gen'] = loss_gen
                logs['loss_disc'] = loss_disc
                logs['loss_matryoshka'] = loss_matryoshka
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                autoencoder.eval()
                state_dict = accelerator.unwrap_model(autoencoder).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"AE-{config.train.exp_name}-{global_step // 1000}k"))

                state_dict = accelerator.unwrap_model(hybrid_loss).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"Loss-{config.train.exp_name}-{global_step // 1000}k"))

                ema_path = os.path.join(output_dir, f"EMA-{config.train.exp_name}-{global_step // 1000}k")
                ema.save_shadow(ema_path)
                # ema.apply_shadow()
                # ema.model.eval()
                # state_dict = ema.model.state_dict()
            accelerator.wait_for_everyone()

            if global_step >= config.train.num_iters:
                training_done = True
                break
        
        epoch += 1

    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ae_total.yaml')
    args = parser.parse_args()
    main(args.config)