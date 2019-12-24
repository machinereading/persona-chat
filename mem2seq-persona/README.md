## This is a (modified) clone of https://github.com/HLTCHKUST/Mem2Seq

### Train Commands
```
python main.py
```
The script trains Mem2Seq model in persona-chat setting, and saves checkpoint in `save_path` which is defined in `config.py`
To change hyper-parameters, modify them in `config.py`
We used following hyper-parameters.
```
train_batch_size: 200
valid_batch_size: 200
hdd: 300
layers: 3
lr: 0.005
dr: 0.2
tr: 0.5
temp: 1.0
position: True
clip: 10.0
epochs: 100
```
