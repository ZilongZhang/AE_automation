import sys
sys.path.insert(1,'../tools')
from transformers import modeling_bert
import numpy as np
import torch
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from tqdm import tqdm
import pdb
import torch.nn.functional as F
from torch import optim
model_name = 'SSI_Bert'
import time
import utils
import imp
#from utils import batch_sent_loader
import os
from sklearn.metrics import confusion_matrix
import json

imp.reload(utils)
hpara1 = utils.hpara()
hpara1.ratio = 2
hpara1.batch_size = 4
hpara1.word_layers=4
hpara1.sent_layers=1
hpara1.use_position_embedding = False
hpara1.word_lr = 1e-04 #5e-05
hpara1.sent_lr = 8e-04 #2e-04
hpara1.decay_step = 10
hpara1.decay_gamma = 0.5
hpara1.max_sent_len = 64
hpara1.fix_cls = True
hpara1.head_num = 12
hpara1.att_decay_step=3
hpara1.att_decay_rate=0.5
hpara1.max_doc_len = 300

hpara1.use_SSI_Bert = True
hpara1.use_narrow = True

pretrain_model_dir = '../BERT/bert/infection_bert_fix_gast'
vocab_file = os.path.join(pretrain_model_dir,'vocab.txt')
bert_config_file = os.path.join(pretrain_model_dir,'bert_config.json')
print(bert_config_file)
training_generator,validation_generator,dummy_generator = utils.load_data_new()
tokenizer = utils._load_tf_tokenizer(vocab_file = vocab_file)

config = modeling_bert.BertConfig.from_json_file(bert_config_file)
config.num_hidden_layers = hpara1.word_layers
config.output_attentions = True
model_word = modeling_bert.BertModel(config)

import re
model_state_dict = pretrain_model_dir + '/pytorch_model.bin'
pretrained_dict = torch.load(model_state_dict)
model_dict = model_word.state_dict()
matched_dict = {}
for k in pretrained_dict.keys():
    try:
        new_k = re.search(r'(bert\.)(.*)',k).group(2)
    except:
        continue
    if new_k in model_dict:
        matched_dict[new_k] = pretrained_dict[k]
model_dict.update(matched_dict)
model_word.load_state_dict(model_dict)
cls_weight = model_word.state_dict()['embeddings.word_embeddings.weight'][101]

config_doc = modeling_bert.BertConfig.from_json_file(bert_config_file)
config_doc.num_hidden_layers = hpara1.sent_layers
config_doc.output_attentions = True
if hpara1.use_narrow:
    config_doc.attention_probs_dropout_prob = 0
config_doc.num_attention_heads = hpara1.head_num
config_doc.attention_probs_dropout_prob = False
use_advanced_loss = hpara1.use_angular
use_position_embedding = hpara1.use_position_embedding
model_sent = modeling_bert.BertModel_no_embedding_narrow(config_doc,cls_weight,use_position_embedding=use_position_embedding,\
                                                        att_decay_rate = hpara1.att_decay_rate,att_decay_step=hpara1.att_decay_step,att_min_sent=hpara1.att_min_sent)
print(config_doc)

model_word = model_word.cuda()
model_sent = model_sent.cuda()

date = str(list(time.localtime())[0:3]).replace(', ','_')
save_dir = './exp/'+model_name+'_'+date[1:-1]+'_narrow0/'

if not os.path.exists(save_dir):
    cmd = 'mkdir -p ' + save_dir
    os.system(cmd)
     
with open(save_dir + 'hpara.json', 'w') as fp:
    json.dump(hpara1.__dict__, fp)
    
do_train = True
do_test = True

max_epoch = hpara1.max_epoch
log = 'Iter {}/{}, Loss={:.3f},accu={:.3f},time={:.3f}\n'
from tqdm import tqdm
batch_size = hpara1.batch_size
accumulation_steps = hpara1.accumulation_steps
max_sent_len = hpara1.max_sent_len
max_doc_len = hpara1.max_doc_len
#progress_bar = tqdm(enumerate(training_generator))
para_dict = {}
hpara_list = []

utils.model_train_and_test(hpara1,model_word,model_sent,save_dir,training_generator,validation_generator,tokenizer=tokenizer,narrow=hpara1.use_narrow,start_epoch = 0)