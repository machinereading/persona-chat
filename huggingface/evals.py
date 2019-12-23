import subprocess
import os
import itertools

#epochs = [80, 70, 60, 50, 40, 30, 20, 10]
epochs = [5, 4, 3]
eval_types = ['hits@1', 'f1']
prod = itertools.product(epochs, eval_types)
template = "python3 convai_evaluation.py --model_checkpoint runs/run_e{:02} --eval_type {} --log_path logs/{}_e{:02}.txt"

#subprocess.run(template.format(80, 'ppl', 'ppl', 80).split())
subprocess.run(template.format(5, 'ppl', 'ppl', 5).split())

for epoch, eval_type in prod:
    command = template.format(epoch, eval_type, eval_type, epoch)
    subprocess.run(command.split())
