## This is a clone of https://github.com/atselousov/transformer_chatbot

### Train Commands
```
python train.py
```
To train from scratch (this initializes weights with pretrained OpenAI GPT), set `load_last = False` of trainer_config in `config.py`
To train from some checkpoint, set `load_last = True` and `trained_checkpoint_path = <the path>` of trainer_config in `config.py`
The trainer loads datasets from paths in `train_datasets (valid_datasets)` of trainer_config in `config.py`

We used following hyper-parameter settings in our re-implementation. (Modify this in `config.py`)
```
n_epochs: 80
batch_size: 160
batch_split: 64
lr: 6.25e-5
lr_warmup: 16000
lm_weight: 0.5
risk_weight: 0
n_jobs: 4
label_smoothing: 0.1
clip_grad: None
test_period: 1
```

### Evaluation Commands
```
python eval_f1.py
python eval_hits.py
```
To evaluate learned model, modify `checkpoint_path` of model_config in `config.py` to the saved checkpoint.
