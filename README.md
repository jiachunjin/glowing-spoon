One click launch on H100:
sh setup_and_launch.sh

```
conda create -n bgpt python=3.8
conda activate bgpt
pip install accelerate==0.33.0 torchvision==0.19.1 webdataset omegaconf einops wandb opencv-python==4.1.2.30
```

if cv2 error:
```
apt-get install libsm6
apt-get install libxrender1
apt-get install libxext-dev
```