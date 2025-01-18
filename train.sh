# accelerate launch --main_process_port 29509 train.py --config 'configs/ae_find_schedule.yaml'
accelerate launch --main_process_port 29508 gpt_train.py
# accelerate launch --main_process_port 29507 train_ae_total.py --config 'configs/ae_total_64.yaml'