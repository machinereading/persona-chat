## This is a clone of https://github.com/atselousov/transformer_chatbot

This project use Python3!
Before use scripts, you should first do following things:

1. Download BPE vocabulary files and a checkpoint file
```
bash envirionment/prepare_environment.sh
```

2. Download datasets
```
python build.py
```

3. Reinstall spacy and Download spacy-en
```
pip uninstall spacy
pip install spacy=2.0.0
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
pip install --user en_core_web_sm-2.0.0.tar.gz
python -m spacy link en_core_web_sm en
```

4. Install dependencies
```
pip install -r requirements.txt
```

5. Download pretrained OpenAI-GPT weights
```
git clone https://github.com/openai/finetune-transformer-lm
mv finetune-transformer-lm/models/* parameters/*
mv parameters/params_shapes.json parameters/parameters_shapes.json
rm -rf finetune-transformer-lm
```

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
