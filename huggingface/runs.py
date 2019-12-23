import subprocess
import os

base_epoch = 20
epochs = [10, 20, 30, 40, 50, 60]
template = "python3 train.py --model_checkpoint runs/run_e{:02} --n_epochs 10 --log_dir runs/run_e{:02}"

for epoch in epochs:
    command = template.format(base_epoch, base_epoch + epoch)
    subprocess.run(command.split())
