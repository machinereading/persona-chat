from attrdict import AttrDict
from model.utils import openai_transformer_config


def get_model_config():
    default_config = openai_transformer_config()
    config = AttrDict({'bpe_vocab_path': '/data/kdgyun425/parameters/bpe.vocab',
                       'bpe_codes_path': '/data/kdgyun425/parameters/bpe.code',
                       'checkpoint_path': '/data/kdgyun425/transformer/checkpoints/last_checkpoint', 
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 256,
                       'beam_size': 3,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': None,
                       'annealing': 0,
                       'length_penalty': 0.6,
                       'n_segments': None})

    return config


def get_trainer_config():
    config = AttrDict({'train': False,
                       'n_epochs': 100,
                       'batch_size': 200,
                       'batch_split': 64,
                       'lr': 6.25e-5,
                       #'lr_warmup': 16000,
                       'lr_warmup': 0,
                       'lm_weight': 0.5,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda:1',
                       'load_last': True, 
                       'load_default': True,
                       'openai_parameters_dir': '/data/kdgyun425/parameters',
                       'trained_checkpoint_path': '/data/kdgyun425/transformer/checkpoints/trained',
                       'default_checkpoint_path': '/data/kdgyun425/transformer/checkpoints/default_checkpoint',
                       'interrupt_checkpoint_path': '/data/kdgyun425/transformer/checkpoints/interrupt_checkpoint',
                       'train_datasets': ['/data/kdgyun425/datasets/ConvAI2/train_self_revised_no_cands.txt',
                                          '/data/kdgyun425/datasets/ConvAI2/train_self_original_no_cands.txt'],
                                          #'./datasets/DailyDialog/train_dailydialog.txt'],
                       'test_datasets': ['/data/kdgyun425/datasets/ConvAI2/valid_self_revised_no_cands.txt',
                                         '/data/kdgyun425/datasets/ConvAI2/valid_self_original_no_cands.txt']})#,
                                         #'./datasets/DailyDialog/valid_dailydialog.txt']})

    return config

