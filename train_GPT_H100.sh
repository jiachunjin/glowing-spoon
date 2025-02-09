conda activate bingpt
# Config accelerate
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

# Login wandb for monitoring the training
echo "Logging into Weights & Biases..."
wandb login 96131f8aede9a09cdcdaecc19c054f804e330d3d

# Launch the training
echo "Launching training..."
accelerate launch --main_process_port 29508 gpt_train.py --config 'configs/H100_GPT_base.yaml'

echo "Training complete!"