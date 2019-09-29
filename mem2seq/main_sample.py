import numpy as np
import logging 
from tqdm import tqdm

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.Mem2Seq import *

'''
python3 main_test.py -dec= -path= -bsz= -ds=
'''

BLEU = False

if (args['decoder'] == "Mem2Seq"):
    if args['dataset']=='kvr':
        from utils.utils_kvr_mem2seq import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi_mem2seq import *
    else: 
        print("You need to provide the --dataset information")
else:
    if args['dataset']=='kvr':
        from utils.utils_kvr import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi import *
    else: 
        print("You need to provide the --dataset information")

# Configure models
directory = args['path'].split("/")[-1]
task = directory.split('HDD')[0]
HDD = directory.split('HDD')[1].split('BSZ')[0]
L = directory.split('L')[1].split('lr')[0]

train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(task, batch_size=int(args['batch']))

if args['decoder'] == "Mem2Seq":
    model = globals()[args['decoder']](
        int(HDD),max_len,max_r,lang,args['path'],task, lr=0.0, n_layers=int(L), dropout=0.0, unk_mask=0)
else:
    model = globals()[args['decoder']](
        int(HDD),max_len,max_r,lang,args['path'],task, lr=0.0, n_layers=int(L), dropout=0.0)

if args['dataset'] == 'kvr':
    with open('data/KVR/kvret_entities.json') as f:
        global_entity = json.load(f)
        global_entity_list = []
        for key in global_entity.keys():
            if key != 'poi':
                global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
            else:
                for item in global_entity['poi']:
                    global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
        global_entity_list = list(set(global_entity_list))
else:
    if int(args["task"])!=6:
        global_entity_list = entityList('data/dialog-bAbI-tasks/dialog-babi-kb-all.txt',int(args["task"]))
    else:
        global_entity_list = entityList('data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt',int(args["task"]))

dev_iter = iter(dev)
data_dev = dev_iter.next()

if args['dataset']=='kvr':
    words = model.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],
                        data_dev[2],data_dev[3],data_dev[4],data_dev[5],data_dev[6]) 
else:
    words = model.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],
            data_dev[2],data_dev[3],data_dev[4],data_dev[5],data_dev[6])

def print_query(seq_list):
    dialog = []
    for seq in seq_list:
        if seq[1] == '$u' or seq[1] == '$s':
            dialog.append(seq[0])
    print('story: ', ' '.join(dialog))
    
def print_answer(seq_list, i):
    answer = []
    for batch in seq_list:
        if batch[i] != '<EOS>':
            answer.append(batch[i])
        else:
            break
    print('answer:', ' '.join(answer))

for i in range(len(data_dev[6])):
    print_query(data_dev[6][i])
    print('target:', data_dev[7][i])
    print_answer(words, i)
    print()
