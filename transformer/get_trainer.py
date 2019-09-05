import torch
from model.utils import load_openai_weights, set_seed
from model.transformer_model import TransformerModel
from model.trainer import Trainer
from model.text import BPEVocab
from model.dataset import FacebookDataset
from config import get_model_config, get_trainer_config


def get_trainer():
    model_config = get_model_config()
    trainer_config = get_trainer_config()

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)

    vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)

    transformer = TransformerModel(n_layers=model_config.n_layers,
                                   n_embeddings=len(vocab),
                                   n_pos_embeddings=model_config.n_pos_embeddings,
                                   embeddings_size=model_config.embeddings_size,
                                   padding_idx=vocab.pad_id,
                                   n_heads=model_config.n_heads,
                                   dropout=model_config.dropout,
                                   embed_dropout=model_config.embed_dropout,
                                   attn_dropout=model_config.attn_dropout,
                                   ff_dropout=model_config.ff_dropout,
                                   bos_id=vocab.bos_id,
                                   eos_id=vocab.eos_id,
                                   max_seq_len=model_config.max_seq_len,
                                   beam_size=model_config.beam_size,  
                                   length_penalty=model_config.length_penalty,
                                   n_segments=model_config.n_segments,
                                   annealing_topk=model_config.annealing_topk,
                                   annealing=model_config.annealing,
                                   diversity_coef=model_config.diversity_coef,
                                   diversity_groups=model_config.diversity_groups)

    if not trainer_config.load_last:
        load_openai_weights(transformer.transformer_module, 
                            trainer_config.openai_parameters_dir,
                            n_special_tokens=vocab.n_special_tokens)
        print('OpenAI weights loaded from {}'.format(trainer_config.openai_parameters_dir))

    train_dataset = FacebookDataset(trainer_config.train_datasets, vocab, transformer.n_pos_embeddings - 1)
    test_dataset = FacebookDataset(trainer_config.test_datasets, vocab, transformer.n_pos_embeddings - 1)

    model_trainer = Trainer(transformer,
                            train_dataset, 
                            test_dataset, 
                            batch_size=trainer_config.batch_size,
                            batch_split=trainer_config.batch_split, 
                            lr=trainer_config.lr, 
                            lr_warmup=trainer_config.lr_warmup, 
                            lm_weight=trainer_config.lm_weight,
                            risk_weight=trainer_config.risk_weight, 
                            n_jobs=trainer_config.n_jobs, 
                            clip_grad=trainer_config.clip_grad, 
                            device=device,
                            ignore_idxs=vocab.special_tokens_ids)

    if trainer_config.load_last:
        state_dict = torch.load(trainer_config.last_checkpoint_path, map_location=device)
        model_trainer.load_state_dict(state_dict)
        print('Weights loaded from {}'.format(trainer_config.last_checkpoint_path))
    
    return model_trainer