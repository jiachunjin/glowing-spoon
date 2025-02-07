def generate_ddp():
    import torch
    from tqdm.auto import tqdm
    from gpt import Transformer_bin
    from ae_total import AE_total
    from omegaconf import OmegaConf
    from accelerate import Accelerator
    from torchvision import transforms

    from generate import generate_blockwise
    from utils import get_latents_mask
    

    accelerator = Accelerator()

    config = OmegaConf.load('configs/a800_GPT.yaml')
    # gpt_ckpt_path = 'pretrained_models/rush1024x32/gpt_40k'
    # cfg_scale = 1 30.490274062402875
    # cfg_scale = 2 10.14147880593265


    # gpt_ckpt_path = 'pretrained_models/gpt_1024x32/gpt_50k'
    # cfg=1: 33.22040093873784, cfg=4: 17.362420300726797, cfg=2: 10.336091123486767
    
    # gpt_ckpt_path = 'pretrained_models/gpt_1024x32/gpt_60k'
    # cfg=1: 31.24576336716774 cfg=2: 9.912285744398048

    # gpt_ckpt_path = 'pretrained_models/gpt_1024x32/gpt_70k'
    # cfg_scale = 1, 29.956380201634204
    # cfg_scale = 2， 9.006749346834283

    # gpt_ckpt_path = 'pretrained_models/gpt_1024x32/gpt_75k'
    # cfg_scale = 1 27.60444920795271
    # cfg_scale = 2 9.408562385452115
    # cfg_scale = 1.75 9.252959067996017
    # cfg_scale = 2.5 11.908764673298833
    # cfg_scale = 1.5 11.162830936462854

    gpt_ckpt_path = 'pretrained_models/gpt_1024x32/gpt_80k'
    # cfg_scale = 1 28.616402281830744
    cfg_scale = 2
    

    gpt = Transformer_bin(config.gpt)
    ckpt = torch.load(gpt_ckpt_path, map_location='cpu', weights_only=True)
    m, u = gpt.load_state_dict(ckpt, strict=False)
    gpt.eval()

    autoencoder = AE_total(config.autoencoder)
    ckpt = torch.load(config.autoencoder.pretrained_ckpt_path, map_location='cpu', weights_only=True)
    autoencoder.load_state_dict(ckpt)
    autoencoder.requires_grad_(False)
    autoencoder.eval()

    autoencoder, gpt = accelerator.prepare(autoencoder, gpt)

    rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes

    labels = torch.arange(rank, 1000, world_size).to(accelerator.device)
    latent_mask = get_latents_mask(
        num_latents = config.autoencoder.num_latents,
        input_dim   = config.autoencoder.binary_dim,
        schedule    = config.autoencoder.decoder_1d.latents_mask_schedule,
    )
    latent_mask = latent_mask.unsqueeze(0).to(accelerator.device)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # seed = 1
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    dtype = torch.float16
    gpt = gpt.to(dtype)


    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage()
    ])
    # generate 50 images for each label
    with torch.no_grad():
        for lable in tqdm(labels):
            cond = torch.tensor([lable]*50).to(accelerator.device)
            with torch.autocast('cuda', enabled=True, dtype=dtype, cache_enabled=True):
                out = generate_blockwise(gpt.module, cond, 1024, cfg_scale, latent_mask, accelerator.device, verbose=False)


            with torch.no_grad(), torch.autocast('cuda', enabled=True, dtype=dtype, cache_enabled=True):
                recon_full = autoencoder.decode_bits(out, num_activated_latent=None)
                for id, rec in enumerate(recon_full):
                    rec = inverse_transform(rec)
                    rec.save(f'assets/gen_total/{rank}_{lable}_{id}.png')
    
    print('done')
    accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    #     import subprocess
    #     command = [
    #         "python", "-m", "pytorch_fid",
    #         "/home/jiachun/codebase/tok/assets/ori",
    #         "/home/jiachun/codebase/tok/assets/gen_total",
    #         "--device", "cuda:7"
    #     ]

    #     # 实时输出日志
    #     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    #     # 遍历输出流
    #     for line in process.stdout:
    #         print(line)
    #     fid = float(line.split()[-1])

    #     # 等待进程结束
    #     process.wait()

    #     print(f'FID: {fid}')

if __name__ == '__main__':
    generate_ddp()

        # z = torch.randn(1, config.autoencoder.latent_dim).to(gpt.device)
        # z = autoencoder.decoder(z)
        # z = z.repeat(world_size, 1)
        # z = z[rank:rank+1]
        # gpt.eval()
        # with torch.no_grad():
        #     samples = gpt.generate_images(z, labels)
        # accelerator.save(samples, f'generated_images/{rank}_{i}.png')
        # print(f'rank {rank} generated {i} images')