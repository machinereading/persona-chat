from tqdm import tqdm
import torch
from Mem2Seq import Mem2Seq
from utils.preprocess import preprocess
from utils.vocab import CustomVocab, merge_bpe
from config import config

device = torch.device(config['device'])
vocab, train_loader, valid_loader, test_loader, max_s, max_r = preprocess(config['vocab_path'], config['codes_path'], config['train_datasets'], config['valid_datasets'], config['test_datasets'], config['train_batch_size'], config['valid_batch_size'], device)


model = Mem2Seq(hidden_size=config['hdd'], n_layers=config['layers'], max_s=max_s, max_r=max_r, vocab=vocab, load_path=config['load_path'], save_path=config['save_path'], lr=config['lr'], dr=config['dr'], position=config['position'], device=device)

print('Version {} - Training starts with model of {} layers, {} hdd, {} lr, {} dr, {} tr, {} temp, {} clip, {} position and {} batch size'.format(config['version'], config['layers'], config['hdd'], config['lr'], config['dr'], config['tr'], config['temp'], config['clip'], config['position'], config['train_batch_size']))

valid_iter = iter(valid_loader)
decay_cnt = 0
decay_num = 0
loss_best = float('inf')
loss_prev = float('inf')

for epoch in range(1, 1 + config['epochs']):  
    # Run the train function
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in pbar: 
        model.train_batch(data, config['clip'], config['tr'], i==0)
        pbar.set_description('<epoch {}> '.format(epoch) + model.print_loss())
       
    loss, bleu, f1, perplexity, perplexity_m = model.evaluate(valid_loader, config['temp'])
    print('BLEU: {:.3f}, F1: {:.3f}, Perplexity: {:.3f}, Masked Perplexity: {:.3f}'.format(bleu, f1, perplexity, perplexity_m))

    if epoch % 10 == 0:
        model.save_model(config['version'], epoch)
        print('<samples>')
        data_dev = valid_iter.next()
        words_batch, _, _, _, _ = model.generate_batch(data_dev, config['temp'])
        for n in range(5):
            print('[targ]: ' + merge_bpe(data_dev[6][n]))
            print('[pred]: ' + merge_bpe(words_batch[n]))
            print()
    
    if loss < loss_best:
        loss_best = loss
        decay_cnt = 0
    
    if loss > loss_prev:
        decay_cnt += 1
        
    loss_prev = loss

    if decay_cnt == 5:
        #model.encoder_scheduler.step()
        #model.decoder_scheduler.step()
        model.scheduler.step()
        decay_num += 1
        decay_cnt = 0
        print('lr decayed')
        
print('Version {} - Training finished with {} epochs, {} lr decays'.format(config['version'], config['epochs'], decay_num))
