#!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
# Exit immediately if a command exits with a non-zero status
set -e
# Step 1: Create and activate conda environment
echo "Creating conda environment..."
conda create -n bgpt python=3.8 -y
. $(conda info --base)/etc/profile.d/conda.sh
conda activate bgpt

# Step 2: Install required packages
echo "Installing required packages..."
pip install accelerate==0.33.0 torchvision==0.19.1 webdataset omegaconf einops wandb opencv-python==4.1.2.30

# Step 3: Download ImageNet dataset (webdataset format)
echo "Downloading ImageNet dataset..."
huggingface-cli download --repo-type dataset --resume-download Cylarus/ImageNet --local-dir ./datasets/imagenet

# Step 4: Download pretrained tokenizer
echo "Downloading on current tokenizer ckpt..."
huggingface-cli download orres/a800_tok_ckpt --local-dir ./ckpts

# Step 5: Confif accelerate
echo "Configuring accelerate (non-interactive)..."
cat <<EOT > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: 0,1,2,3,4,5,6,7
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOT

# Step 6: Login wandb for monitoring the training
echo "Logging into Weights & Biases..."
wandb login 96131f8aede9a09cdcdaecc19c054f804e330d3d

# Step 7: Launch the training
echo "Launching training..."
sh train_tok_h100.sh

echo "Training complete!"