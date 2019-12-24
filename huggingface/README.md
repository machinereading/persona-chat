## This is a clone of https://github.com/huggingface/transfer-learning-conv-ai

### Train Command
Following command trains model `n_epochs` epochs, initialized with pretrained OpenAI-GPT, and saves checkpoint in `log_dir`.
```
python train.py --n_epochs 20 --model_checkpoint openai-gpt --log_dir runs/run_e20
```

### Hyper-Parameter Settings
In our re-implementation, we used following hyper-parameters.
```
num_candidates: 2
max_history: 2
train_batch_size: 4
valid_batch_size: 4
gradient_accumulation_steps: 8
lr: 6.25e-5
lm_coef: 1.0
mc_coef: 1.0
max_norm: 1.0
n_epochs: 20
personality_permutations: 1
```

### Evaluation Command
Each commands evaluates learned model using F1-score, hits@1 and perplexity.
'''
python eval.py --eval_type f1 --model_checkpoint runs/run_e20
python eval.py --eval_type hits@1 --model_checkpoint runs/run_e20
python eval.py --eval_type ppl --model_checkpoint runs/run_e20
'''

### Sample Command
We implemented sampling script to sample utterances from learned model.
It creates `n_samples` samples, iterating from the beginning of the dataset with step `sample_term`.
'''
python sample.py --model_checkpoint runs/run_e20 --n_samples 10 --sample_term 1
'''

### Interaction Command
This command is to conversate interactively with the learned model.
```
python interact.py --model_checkpoint runs/run_e20 --temperature 0.7 --top_p 0.9 --max_length 20
```
