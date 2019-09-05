import logging
from tqdm import tqdm
import torch
from Mem2Seq import Mem2Seq
from utils.preprocess import preprocess
from utils.vocab import CustomVocab

def merge_bpe(words):
        for i, word in enumerate(words):
            if word == '<eos>' and i < len(words) - 1:
                words = words[:i+1]
        return ''.join(words).replace(CustomVocab.we, ' ')

vocab_path = './parameters/bpe.vocab'
codes_path = './parameters/bpe.code'

train_datasets = ['./datasets/train_self_revised_no_cands.txt',
                  './datasets/train_self_original_no_cands.txt']
valid_datasets = ['./datasets/valid_self_revised_no_cands.txt']
test_datasets = ['./datasets/valid_self_original_no_cands.txt']

batch_size = 32
device = torch.device('cuda:0')

vocab, train_loader, valid_loader, test_loader, max_r = preprocess(vocab_path, codes_path, train_datasets, valid_datasets, test_datasets, batch_size, device)

lr = 0.005
hdd = 300
pos = 512
layer = 4
dr = 0.2
#path = 'save/HDD300POS512DR0.2L4lr0.005v2'
path = None
model = Mem2Seq(hidden_size=hdd, pos_size=pos, max_r=max_r, vocab=vocab, path=path,
                lr=lr, n_layers=layer, dropout=dr, unk_mask=True, device=device)
valid_iter = iter(valid_loader)

valid_cnt = 0
valid_best = 0.0
for epoch in range(1, 101):
    logging.info("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in pbar: 
        model.train_batch(data[0], data[1], data[2], data[3], data[4], len(data[1]), 10.0, 0.5, i==0)
        pbar.set_description('<epoch {}> '.format(epoch) + model.print_loss())
   
    if epoch % 10 == 0:
        print('<samples>')
        data_dev = valid_iter.next()
        words_batch = model.generate_batch(batch_size, data_dev[0], data_dev[1], data_dev[5])
        for n in range(3):
            print('[targ]: ' + merge_bpe(data_dev[6][n]))
            print('[pred]: ' + merge_bpe(words_batch[n]))
            print()
    
    valid_bleu = model.evaluate(valid_loader)

    if valid_bleu >= valid_best - 0.05:
        valid_best = valid_bleu
        valid_cnt = 0
        model.save_model(version=2)
    else:
        valid_cnt += 1

    if valid_cnt == 5:
        model.scheduler.step(model.loss)

    if valid_cnt == 10:
        print('early stopped!')
        break
        

test_bleu = model.evaluate(test_loader)
print('test bleu: {:.3f}'.format(test_bleu))
