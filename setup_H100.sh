#!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
# Exit immediately if a command exits with a non-zero status
set -e
# Step 1: Create and activate conda environment
echo "Creating conda environment..."
conda create -n bingpt python=3.8 -y
. $(conda info --base)/etc/profile.d/conda.sh
conda activate bingpt

# Step 2: Install required packages
echo "Installing required packages..."
python3 -m pip install accelerate==0.33.0 torchvision==0.19.1 webdataset omegaconf einops wandb opencv-python==4.1.2.30 pytorch-fid

# Step 3: Download ImageNet dataset (webdataset format)
echo "Downloading ImageNet dataset..."
huggingface-cli download --repo-type dataset --resume-download Cylarus/ImageNet --local-dir /checkpoint/data_gen_exp/datasets/imagenet

# Step 4: Download pretrained tokenizer
echo "Downloading pretrained tokenizer ckpts..."
huggingface-cli download orres/H100_ckpts --local-dir /checkpoint/data_gen_exp/pretrained_ckpts

# Step 5: Download validation image
echo "Downloading validation images..."
huggingface-cli download orres/imagenet_val --local-dir /checkpoint/data_gen_exp/eval_images/
tar -xvf /checkpoint/data_gen_exp/eval_images/assets_ori.tar -C /checkpoint/data_gen_exp/eval_images
mkdir /checkpoint/data_gen_exp/eval_images/assets/gen

echo "Setup done, ready to train!"