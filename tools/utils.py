import pickle
import torch
import pdb
from torch import optim
from sklearn.metrics import confusion_matrix,roc_auc_score
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tokenization
import time
import re
import json
from functools import partial
from transformers import modeling_bert
from collections import defaultdict
from multiprocessing import Pool
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax

class DocDataset(Dataset):
    def __init__(self, Doc_list, labels):
        #'Initialization'
        self.labels = labels
        self.Doc_list = Doc_list
    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.Doc_list)
    def __getitem__(self, index):
        #'Generates one sample of data'
        # Load data and get label
        X = self.Doc_list[index]
        y = self.labels[index]
        return X, y
'''
class DocDataset_with_index(Dataset):
    def __init__(self, Doc_list, labels):
        #'Initialization'
        self.labels = labels
        self.Doc_list = Doc_list
    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.Doc_list)
    def __getitem__(self, index):
        #'Generates one sample of data'
        # Load data and get label
        X = self.Doc_list[index]
        y = self.labels[index]
        return X,y,index
'''        
def load_data(ratio=2,clarity=False,stratify = True,shuffle_train = True,corr_label=True):
    src_dir = 'pu_data_ratio' + str(ratio) + '/'
    if clarity:
        src_dir = 'pu_data_ratio_Clarity' + str(ratio) + '/'
    if stratify:
        text_train = pickle.load(open(src_dir+'text_train_stratified.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train_stratified.pkl','rb'))
        text_test = pickle.load(open(src_dir+'text_test_stratified.pkl','rb'))
        if corr_label and os.path.exists(src_dir+'corrected_label_test_stratify.pkl'):
            label_test = pickle.load(open(src_dir+'corrected_label_test_stratify.pkl','rb'))
        else:
            label_test = pickle.load(open(src_dir+'label_test_stratified.pkl','rb'))
    else:
        text_train = pickle.load(open(src_dir+'text_train.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train.pkl','rb'))
        text_test = pickle.load(open(src_dir+'text_test.pkl','rb'))
        label_test = pickle.load(open(src_dir+'label_test.pkl','rb'))
    params_sent = {'batch_size': 1,
            'shuffle': shuffle_train,
            'num_workers': 6}
    params_sent_validation = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 6}
    training_set = DocDataset(text_train, label_train)
    training_generator = DataLoader(training_set, **params_sent)
    validation_set = DocDataset(text_test, label_test)
    validation_generator =DataLoader(validation_set, **params_sent_validation)
    dummy_set = DocDataset(text_train[:5], label_train[:5])
    dummy_generator = DataLoader(dummy_set, **params_sent)
    return training_generator,validation_generator,dummy_generator
# Convert splited sentence to documents. 

def load_data_new(shuffle_train = True,corr_label=True,merged=True):
    src_dir = '../preprocessing/result/ratio2/'
    if merged:
        text_train = pickle.load(open(src_dir+'merged_text_train.pkl','rb'))
        text_test = pickle.load(open(src_dir+'merged_text_test.pkl','rb'))
    else:
        text_train = pickle.load(open(src_dir+'text_train.pkl','rb'))
        text_test = pickle.load(open(src_dir+'text_test.pkl','rb'))
    
    if corr_label and os.path.exists(src_dir+'corrected_label_test.pkl'):
        print('Loaded the corrected labels')
        label_test = pickle.load(open(src_dir+'corrected_label_test.pkl','rb'))
        label_train = pickle.load(open(src_dir+'corrected_label_train.pkl','rb'))        
    else:
        label_test = pickle.load(open(src_dir+'label_test.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train.pkl','rb'))
    
        
    params_sent = {'batch_size': 1,
        'shuffle': shuffle_train,
        'num_workers': 6}
    params_sent_validation = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 6}
    training_set = DocDataset(text_train, label_train)
    training_generator = DataLoader(training_set, **params_sent)

    validation_set = DocDataset(text_test, label_test)
    validation_generator =DataLoader(validation_set, **params_sent_validation)
    dummy_set = DocDataset(text_train[:5], label_train[:5])
    dummy_generator = DataLoader(dummy_set, **params_sent)
    return training_generator,validation_generator,dummy_generator
  
def load_data_for_correction():
    src_dir = '../preprocessing/result/ratio2/'
    text_train = pickle.load(open(src_dir+'merged_text_train.pkl','rb'))
    text_test = pickle.load(open(src_dir+'merged_text_test.pkl','rb')) 
    print(len(text_train))
    text_train.extend(text_test)
    
    label_train = pickle.load(open(src_dir+'label_train.pkl','rb'))    
    label_test = pickle.load(open(src_dir+'label_test.pkl','rb'))
    label_train.extend(label_test)
    params_all = {'batch_size': 1,
        'shuffle': False,
        'num_workers': 6}
    all_set = DocDataset(text_train, label_train)
    all_generator =DataLoader(all_set, **params_all)
    return all_generator
    
def load_review_data(select_index,corr_label=True):
    src_dir = '../preprocessing/result/ratio2/'
    text_test = pickle.load(open(src_dir+'text_test.pkl','rb'))
    select_text_test = []
    select_label_test = []
    if corr_label and os.path.exists(src_dir+'corrected_label_test.pkl'):
        print('Loaded the corrected labels')
        label_test = pickle.load(open(src_dir+'corrected_label_test.pkl','rb'))
    else:
        label_test = pickle.load(open(src_dir+'label_test.pkl','rb'))
    for i in select_index:
        select_text_test.append(text_test[i])
        select_label_test.append(label_test[i])
    params_sent_validation = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 6}
    validation_set = DocDataset(select_text_test, select_label_test)
    validation_generator =DataLoader(validation_set, **params_sent_validation)
    return validation_generator

def load_raw_data(ratio=2,clarity=False,stratify = True,corr_label=True):
    src_dir = 'pu_data_ratio' + str(ratio) + '/'
    if clarity:
        src_dir = 'pu_data_ratio_Clarity' + str(ratio) + '/'
    if stratify:
        text_train = pickle.load(open(src_dir+'text_train_stratified.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train_stratified.pkl','rb'))
        text_test = pickle.load(open(src_dir+'text_test_stratified.pkl','rb'))
        if corr_label and os.path.exists(src_dir+'corrected_label_test_stratify.pkl'):
            label_test = pickle.load(open(src_dir+'corrected_label_test_stratify.pkl','rb'))
        else:
            label_test = pickle.load(open(src_dir+'label_test_stratified.pkl','rb'))
    else:
        text_train = pickle.load(open(src_dir+'text_train.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train.pkl','rb'))
        text_test = pickle.load(open(src_dir+'text_test.pkl','rb'))
        label_test = pickle.load(open(src_dir+'label_test.pkl','rb'))
    return text_train,label_train,text_test,label_test

def load_raw_data_new(corr_label=True):
    src_dir = '../preprocessing/result/ratio2/'
    text_train = pickle.load(open(src_dir+'text_train.pkl','rb'))
    text_test = pickle.load(open(src_dir+'text_test.pkl','rb'))
    if corr_label and os.path.exists(src_dir+'corrected_label_test.pkl'):
        label_test = pickle.load(open(src_dir+'corrected_label_test.pkl','rb'))
        label_train = pickle.load(open(src_dir+'corrected_label_train.pkl','rb'))        
    else:
        label_test = pickle.load(open(src_dir+'label_test.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train.pkl','rb'))
    return text_train,label_train,text_test,label_test
    
def back_to_doc(doc_list):
    new_doc_list = []
    for doc in doc_list:
        new_doc_list.append('')
        for sent in doc:
            new_doc_list[-1]+=' ' + str(sent)
    return new_doc_list
    
def cancat_to_20(sent_list):
    extended_sent_list = []
    sent_index = 0
    while sent_index < len(sent_list):
        tmp_long_sent = ''
        tmp_word_num = 0
        while tmp_word_num < 20 and sent_index < len(sent_list):
            tmp_long_sent += sent_list[sent_index] +' '
            tmp_word_num += len(sent_list[sent_index].split())
            sent_index +=1
        extended_sent_list.append(tmp_long_sent)
    extended_sent_list.append(tmp_long_sent)
    return extended_sent_list
        
def read_vocab_dict(vocab_file):
    vocab_dict = {}
    index = 0
    with open(vocab_file) as f:
        for line in f.readlines():
            if line:
                vocab_dict[index] = line.strip()
                index +=1
    return vocab_dict
    
def wrong_statis(y,y_hat,statis_dict):
    y = np.array(y)
    y_hat = np.array(y_hat)
    diff = y - y_hat
    false_positive_indices = np.where(diff == -1)[0]
    false_negative_indices = np.where(diff ==  1)[0]
    if np.sum(y_hat):
        statis_dict['false_positive'].append(false_positive_indices)
        statis_dict['false_negative'].append(false_negative_indices)
    return statis_dict

def dir_model_iter(src_dir):
    index_list = []
    for file in os.listdir(src_dir):
        if 'save_word' in file:
            cur_index = re.search('\d+',file).group()
            index_list.append(int(cur_index))
    index_list.sort()
    for index in index_list:
        word_model = src_dir +'/save_word_' + str(index) + '.bin'
        sent_model = src_dir +'/save_sent_' + str(index) + '.bin'
        yield word_model,sent_model
        
def _load_tf_tokenizer(vocab_file=None,uncased=True):
    if vocab_file is None:
        vocab_file ='/gpfs/qlong/home/tzzhang/mimicIII/nn_code/biobert_pretrain_output_all_notes_150000/vocab.txt'
    tf_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=uncased)
    return tf_tokenizer

def batch_sent_loader(doc_text,batch_size,batch_count,max_doc_len=400):
    start = batch_size * batch_count 
    upper_bound = min(len(doc_text),max_doc_len-1)
    end = min(upper_bound, batch_size * (batch_count+1))
    batch_sent = doc_text[start:end]
    return batch_sent

def word_tokenize(text,max_seq_length,tokenizer=None):
    #print(text)
    if tokenizer is None:
        tokenizer = _load_tf_tokenizer()
    text = text.replace('\n',' ')
    raw_tokens = tokenizer.tokenize(text)
    if len(raw_tokens) > max_seq_length - 2:
        raw_tokens = raw_tokens[0:(max_seq_length - 2)]
    tokens = []
    tokens.append("[CLS]")
    for token in raw_tokens:
        tokens.append(token)
    tokens.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
    return input_ids,input_mask
            
class hpara:
    def __init__(self):
        self.word_lr = 0.00005
        self.sent_lr = 0.0002
        self.max_epoch = 60
        self.batch_size = 256
        self.accumulation_steps = 40
        self.max_sent_len = 64
        self.max_doc_len = 400
        self.ratio = 2
        self.word_layers = 6
        self.sent_layers = 3
        self.hidden_size = 768
        self.use_angular = False
        self.use_PU_Bert = True
        self.use_narrow = True
        self.weight = 1
        self.decay_step = 10
        self.decay_gamma = 0.5
        self.att_decay_step = 3
        self.att_decay_rate = 0.5
        self.att_min_sent = 10

    
def model_train_and_test(hpara1,model_word,model_sent,save_dir,\
                        training_generator,validation_generator, \
                        tokenizer,do_train=True,do_test=True,pos_loss_weight = 1,\
                        decay_step = 10,decay_gamma = 0.5,narrow=False,start_epoch = 0):
    # Set up hyper parameters
    loss_weight = torch.tensor([1,pos_loss_weight]).float().cuda()
    criterion = torch.nn.NLLLoss(weight = loss_weight)
    log_file = open(save_dir+'log','a')
    max_epoch = hpara1.max_epoch
    log = 'Iter {}/{}, Loss={:.3f},accu={:.3f},time={:.3f}\n'
    test_log = 'Iter {}/{}, Loss={:.3f},accu={:.3f},auc={:.3f},time={:.3f}\n'
    batch_size = hpara1.batch_size
    accumulation_steps = hpara1.accumulation_steps
    max_sent_len = hpara1.max_sent_len
    max_doc_len = hpara1.max_doc_len
    #progress_bar = tqdm(enumerate(training_generator))
    optimizer = optim.Adamax
    word_optimizer = optimizer(model_word.parameters(), lr=hpara1.word_lr)
    sent_optimizer = optimizer(model_sent.parameters(), lr=hpara1.sent_lr)
    word_scheduler = optim.lr_scheduler.StepLR(word_optimizer, step_size=decay_step, gamma=0.5)
    sent_scheduler = optim.lr_scheduler.StepLR(sent_optimizer, step_size=decay_step, gamma=0.5)
    para_dict = {}
    hpara_list = []
    #tokenizer = _load_tf_tokenizer(vocab_file = '/gpfs/qlong/home/tzzhang/nlp_test/bert/mimic_based_complete_model/vocab.txt')
    #cls_weight = torch.tensor(np.load('cls_weight.npy')).cuda()
    #sep_weight = torch.tensor(np.load('sep_weight.npy')).cuda()
    # do actual training and testing

    for cur_epoch in range(start_epoch,max_epoch):
        log_file = open(save_dir+'log','a')
        if do_train:
            start = time.time()
            model_sent.train()
            model_word.train()
            correct = sum_loss =total_num = 0
            for doc_count,(doc,label) in enumerate(training_generator):
                total_num += len(label)
                if doc_count%1000 ==0:
                    print(doc_count)
                label = label.cuda()
                batch_count=0
                sent_num=0
                end_ind=0
                input_tensors = torch.zeros([1,max_doc_len,hpara1.hidden_size]).cuda()
                # Add cls in sent level
                #input_tensors[0,0] = cls_weight
                while end_ind <= len(doc) and end_ind < max_doc_len-1:
                    batch_sent = batch_sent_loader(doc,batch_size,batch_count,max_doc_len=max_doc_len)
                    cur_batch_size = len(batch_sent)
                    sent_num += cur_batch_size
                    input_ids = torch.zeros(cur_batch_size,max_sent_len).long().cuda()
                    input_mask = torch.ones(cur_batch_size,max_sent_len).long().cuda()
                    for i in range(len(batch_sent)):
                        tmp_ids,tmp_mask = word_tokenize(batch_sent[i][0],max_sent_len,tokenizer)
                        input_ids[i,:] = torch.tensor(tmp_ids)
                        input_mask[i,:] = torch.tensor(tmp_mask)
                    #pdb.set_trace()
                    _,tmp_input_tensors,word_att_output = model_word(input_ids,attention_mask=input_mask)
                    start_ind = batch_count*batch_size + 1 # because the cls was added in 0-th raw 
                    end_ind = start_ind + cur_batch_size
                    input_tensors[0,start_ind:end_ind] = tmp_input_tensors
                    batch_count +=1
                # -----------Add sep in sent matrix----------
                '''
                if end_ind<max_doc_len-1:
                    input_tensors[0,end_ind] = sep_weight
                else:
                    end_ind = max_doc_len-2
                    input_tensors[0,end_ind] = sep_weight
                '''
                #input_tensors[0,end_ind] = sep_weight
                #sent_mask = [1]*(end_ind+1)
                sent_mask = [1]*(end_ind)
                while len(sent_mask)<max_doc_len:
                    sent_mask.append(0)
                sent_mask = torch.tensor(sent_mask).unsqueeze(0).cuda()
                if narrow:
                    _,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask,epoch=cur_epoch)
                else:
                    #import pdb;pdb.set_trace()
                    _,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
                loss = criterion(proba, label)
                #pdb.set_trace()
                sum_loss += loss.item() * len(label)
                _,predicted = torch.max(proba,1)
                correct += (predicted == label).sum()
                loss = loss / accumulation_steps                # Normalize our loss (if averaged)
                loss.backward()                                 # Backward pass
                if (doc_count+1) % accumulation_steps == 0:             # Wait for several backward steps
                    sent_optimizer.step()                           # Now we can do an optimizer step
                    word_optimizer.step()
                    word_optimizer.zero_grad()                           # Reset gradients tensors
                    sent_optimizer.zero_grad()

            torch.save(model_word.state_dict(),save_dir+'/save_word_'+str(cur_epoch)+'.bin')
            torch.save(model_sent.state_dict(),save_dir+'/save_sent_'+str(cur_epoch)+'.bin')
            accu = correct.item() / total_num
            to_print = log.format(cur_epoch,max_epoch,sum_loss,accu,time.time() - start)
            print(to_print)
            log_file.writelines(to_print)
            word_scheduler.step()
            sent_scheduler.step()
        if do_test:
            start = time.time()
            model_sent.eval()
            model_word.eval()
            pred_list = []
            y_list = []
            y_hat = []
            correct = sum_loss =total_num = 0
            for doc_count,(doc,label) in enumerate(validation_generator):
                total_num += len(label)
                label = label.cuda()
                batch_count=0
                sent_num = 0
                end_ind = 0
                input_tensors = torch.zeros([1,max_doc_len,hpara1.hidden_size]).cuda()
                # Add cls in sent level
                #input_tensors[0,0] = cls_weight
                while end_ind <= len(doc) and end_ind < max_doc_len-1:
                    batch_sent =batch_sent_loader(doc,batch_size,batch_count,max_doc_len=max_doc_len)
                    cur_batch_size = len(batch_sent)
                    sent_num += cur_batch_size
                    input_ids = torch.zeros(cur_batch_size,max_sent_len).long().cuda()
                    input_mask = torch.ones(cur_batch_size,max_sent_len).long().cuda()
                    for i in range(len(batch_sent)):
                        tmp_ids,tmp_mask= word_tokenize(batch_sent[i][0],max_sent_len,tokenizer)
                        input_ids[i,:] = torch.tensor(tmp_ids)
                        input_mask[i,:] = torch.tensor(tmp_mask)
                    _,tmp_input_tensors,word_att_output = model_word(input_ids,attention_mask=input_mask)
                    start_ind = batch_count*batch_size+1
                    end_ind = start_ind + cur_batch_size
                    input_tensors[0,start_ind:end_ind] = tmp_input_tensors
                    batch_count +=1
                # -----------Add sep in sent matrix----------
                '''
                if end_ind<max_doc_len:
                    input_tensors[0,end_ind] = sep_weight
                else:
                    end_ind = max_doc_len-1
                    input_tensors[0,end_ind] = sep_weight
                '''
                #input_tensors[0,end_ind] = sep_weight
                #sent_mask = [1]*(end_ind+1)
                sent_mask = [1]*(end_ind)
                while len(sent_mask)<max_doc_len:
                    sent_mask.append(0)
                sent_mask = torch.tensor(sent_mask).unsqueeze(0).cuda()
                if hpara1.use_angular:
                    loss,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
                else:
                    if narrow:
                        _,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask,epoch=cur_epoch)
                    else:
                        _,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
                    loss = criterion(proba, label)            
                pos_socre = np.exp(proba.cpu().detach().numpy())[0,1]
                y_hat.append(pos_socre)
                loss = criterion(proba, label)
                _,predicted = torch.max(proba,1)
                #for making confusion matrix
                pred_list.append(predicted.item())
                y_list.append(label.item())

                sum_loss += loss.item() * len(label)
                #print(logits)
                correct += (predicted == label).sum()
            accu = correct.item() / total_num
            roc_score = roc_auc_score(y_list, y_hat)
            to_print=test_log.format(cur_epoch,max_epoch,sum_loss,accu,roc_score,time.time() - start)
            to_print = 'Test ' + to_print
            print(to_print)
            print(confusion_matrix(y_list,pred_list))
            log_file.writelines(to_print)
            log_file.close()
            if not do_train:
                break

def f1_maximize(y_pred,y):
    y = np.array(y)
    optimal_point = {'max_f1':0,'precision':0,'recall':0,'thres':0}
    f1_list = []
    for thres in np.arange(0,1,0.01):
        round_y_pred = y_pred > thres
        round_y_pred = np.array(round_y_pred)
        tp = np.matmul(round_y_pred,y)
        tn = np.matmul(1-round_y_pred,1-y)
        fp = np.matmul(round_y_pred,1-y)
        fn = np.matmul(1-round_y_pred,y)
        precision = tp / (np.sum([tp,fp]) + 0.00001)
        recall = tp / (np.sum([tp,fn])+ 0.00001)
        f1_score = 2 * precision * recall / (precision + recall)
        f1_list.append(f1_score)       
        if f1_score > optimal_point['max_f1']:
            optimal_point['max_f1'] = f1_score
            optimal_point['precision'] = precision
            optimal_point['recall'] = recall
            optimal_point['thres'] = thres
    round_y_pred = y_pred > optimal_point['thres']
    print(confusion_matrix(y,round_y_pred))
    return optimal_point,f1_list
    
def _detailed_att(token,sent_att,word_att): # Only for one layer Bert
    total_word_att = word_att[0][0]
    for i in range(1,len(word_att)):
        total_word_att = torch.cat((total_word_att,word_att[i][0]),0)
        
def output_att_scores(hpara1,model_word,model_sent,save_dir,\
                       validation_generator,tokenizer,use_angular=False,narrow=False,epoch=0):               
    # Set up hyper parameters
    criterion = torch.nn.NLLLoss()
    log_file = open(save_dir+'log','a')
    max_epoch = hpara1.max_epoch
    log = 'Iter {}/{}, Loss={:.3f},accu={:.3f},time={:.3f}\n'
    test_log = 'Iter {}/{}, Loss={:.3f},accu={:.3f},auc={:.3f},time={:.3f}\n'
    batch_size = hpara1.batch_size
    accumulation_steps = hpara1.accumulation_steps
    max_sent_len = hpara1.max_sent_len
    max_doc_len = hpara1.max_doc_len
    #progress_bar = tqdm(enumerate(training_generator))
    optimizer = optim.Adamax
    word_optimizer = optimizer(model_word.parameters(), lr=hpara1.word_lr)
    sent_optimizer = optimizer(model_sent.parameters(), lr=hpara1.sent_lr)
    para_dict = {}
    hpara_list = []
    #tokenizer = _load_tf_tokenizer(vocab_file = '/gpfs/qlong/home/tzzhang/nlp_test/bert/mimic_based_complete_model/vocab.txt')
    #cls_weight = torch.tensor(np.load('cls_weight.npy')).cuda()
    #sep_weight = torch.tensor(np.load('sep_weight.npy')).cuda()
    model_sent.eval()
    model_word.eval()
    pred_list = []
    y_list = []
    y_hat = []
    correct = sum_loss =total_num = 0
    for doc_count,(doc,label) in enumerate(validation_generator):
        total_num += len(label)
        label = label.cuda()
        batch_count=0
        sent_num = 0
        end_ind = 0
        y_list = []
        y_hat = []
        input_tensors = torch.zeros([1,max_doc_len,hpara1.hidden_size]).cuda()
        # Add cls in sent level
        #input_tensors[0,0] = cls_weight
        word_att_list = []
        while end_ind <= len(doc) and end_ind < max_doc_len-1:
            batch_sent =batch_sent_loader(doc,batch_size,batch_count,max_doc_len=max_doc_len)
            cur_batch_size = len(batch_sent)
            sent_num += cur_batch_size
            input_ids = torch.zeros(cur_batch_size,max_sent_len).long().cuda()
            input_mask = torch.ones(cur_batch_size,max_sent_len).long().cuda()
            for i in range(len(batch_sent)):
                tmp_ids,tmp_mask= word_tokenize(batch_sent[i][0],max_sent_len,tokenizer)
                input_ids[i,:] = torch.tensor(tmp_ids)
                input_mask[i,:] = torch.tensor(tmp_mask)
            _,tmp_input_tensors,word_att_output = model_word(input_ids,attention_mask=input_mask)
            #pdb.set_trace()
            word_att_list.append(word_att_output)
            start_ind = batch_count*batch_size+1
            end_ind = start_ind + cur_batch_size
            input_tensors[0,start_ind:end_ind] = tmp_input_tensors
            batch_count +=1
        #input_tensors[0,end_ind] = sep_weight
        #sent_mask = [1]*(end_ind+1)
        sent_mask = [1]*(end_ind)
        while len(sent_mask)<max_doc_len:
            sent_mask.append(0)
        sent_mask = torch.tensor(sent_mask).unsqueeze(0).cuda()
        if hpara1.use_angular:
            loss,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
        else:
            if narrow:
                _,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask,epoch = epoch)
            else:
                _,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
            loss = criterion(proba, label)
        yield word_att_list,sent_att_output,doc,torch.exp(proba),label
       
from scipy.special import softmax
def multi_layer_back_trace(att_heads): # need a tensor as input
    seq_len = att_heads[0].shape[2]
    layer_num = len(att_heads)
    last_layer_portion = np.diag([1] * seq_len)
    for i in range(layer_num):
        one_layer_att_heads = att_heads[i][0]
        sum_through_heads = torch.sum(one_layer_att_heads,0)
        sum_through_heads = sum_through_heads.cpu().detach().numpy()
        sum_through_heads = np.where(sum_through_heads > 0.001, sum_through_heads, sum_through_heads -1000)
        prop_through_heads = softmax(sum_through_heads,axis=1)
        curr_layer_portion = np.matmul(prop_through_heads, last_layer_portion)
    return curr_layer_portion

def SEP_filter_out(doc,sent_weight):
    be_filtered = False
    if np.argmax(sent_weight[0,:]) == len(doc) + 1:
        if np.max(sent_weight[0,:]) > 0.9:
            be_filtered = True
    return be_filtered

class INATTENTION_COUNT():
    def __init__(self):
        self.total_inatt = 0
        self.pos_inatt = 0
        self.neg_inatt = 0
        
    def count(self,doc,sent_weight,label):
        is_inatt = False
        if np.argmax(sent_weight[0,:]) == len(doc) + 1 or np.argmax(sent_weight[0,:]) == 0:
            self.total_inatt+=1
            if label:
                self.pos_inatt += 1
            else:
                self.neg_inatt += 1
            is_inatt = True
        return is_inatt
        
def write_attn_to_file(attn_generator,des_dir,thres=0.5):
    index =0
    attn_file = os.path.join(des_dir,'attn_log')
    fn_file = os.path.join(des_dir,'fn_log')
    fp_file = os.path.join(des_dir,'fp_log')
    log = 'Doc {}, Predicted proba: {} target:{} \nkey sentence1: {}, score: {},\nkey sentence2: {}, score: {},\nkey sentence3: {}, score: {}\n\n'
    log_without_sent = 'Doc {}, Predicted proba: {} target:{}\n\n' 
    y_list = []
    y_hat = []
    pred_list = []
    for _,sent_att_output,doc,proba,label in attn_generator:
        y_list.append(label.item())
        write_file = open(attn_file,'a')
        proba = proba.cpu().detach().numpy()
        y_hat.append(proba[0,1])
        pred_list.append(proba[0,1]>thres)
        is_wrong = (proba[0,1]>thres) - label.cpu().numpy()
        sent_weight = One_layer_back_trace(sent_att_output[-1])
        #key_sent_index = np.where(sent_weight[0,:]>0.001)[0]
        key_sent_index = np.where(sent_weight[0,1:-1]>0.001)[0] + 1
        key_sent_index = key_sent_index[np.argsort(sent_weight[0,key_sent_index])].tolist()
        doc.append('SEP')
        doc.insert(0,'CLS')        
        #pdb.set_trace()
        if len(key_sent_index) == 0:
            key_sent_index = [np.argmax(sent_weight[0,:])]
        if len(key_sent_index) <3:
            [key_sent_index.append(key_sent_index[0]) for i in range(3 - len(key_sent_index))]
        
        sent1 = doc[key_sent_index[-1]]
        weight1 = sent_weight[0,key_sent_index[-1]]
        
        sent2 = doc[key_sent_index[-2]]
        weight2 = sent_weight[0,key_sent_index[-2]]
        
        sent3 = doc[key_sent_index[-3]]
        weight3 = sent_weight[0,key_sent_index[-3]]
        
        to_print = log.format(index,proba,label,sent1,weight1,sent2,weight2,sent3,weight3)
        write_file.writelines(to_print)
        if is_wrong:
            if is_wrong == 1: # predicted 1 but label was 0
                with open(fp_file,'a') as f:
                    f.writelines(to_print)
            else: #predicted 0 but label was 1
                with open(fn_file,'a') as f:
                    f.writelines(to_print)
        index += 1
        if not index % 100:
            print(index)
        write_file.close()
    print(confusion_matrix(y_list,pred_list))
    return y_list,pred_list,y_hat

class one_ds():
    def __init__(self,SUBJECT_ID,HADM_ID,index):
        self.subject_id = SUBJECT_ID
        self.hadm_id = HADM_ID
        self.index = index
        
class MapBack():
    def __init__(self,src_csv):
        self.data = pd.read_csv(src_csv)
        self.text = self.data['TEXT'].to_list()
    def search(self,sent):
        for doc in self.text[:60415]:
            doc_no_change_line = doc.replace('\n',' ')
            if sent in doc_no_change_line:
                return doc
        return 0
    def search_index(self,sent):
        for i in range(len(self.text[:60415])):
            doc_no_change_line = self.text[i].replace('\n',' ')
            #doc_no_change_line = re.sub(r'\s*\[.*?\]\s*',' ',doc_no_change_line)
            if sent in doc_no_change_line:
                return i
        return 0
        
    def search_more(self,sent):
        ds_dict = {'subject_id':0,'hadm_id':0,'index':0}
        for i in range(len(self.text[60415])):
            doc_no_change_line = self.text[i].replace('\n',' ')
            doc_no_change_line = re.sub(r'\s*\[.*?\]\s*',' ',doc_no_change_line)
            if sent in doc_no_change_line:
                ds_dict['subject_id'] = int(self.data.loc[i]['SUBJECT_ID'])
                ds_dict['hadm_id'] = int(self.data.loc[i]['HADM_ID'])
                ds_dict['index'] = i
                ds_dict['text'] = self.text[i]
        return ds_dict
            
        
        
def get_ids_to_words_dict(vocab_file =None):
    if vocab_file == None:
        vocab_file ='/gpfs/qlong/home/tzzhang/nlp_test/bert/mimic_based_complete_model/vocab.txt'
    dict = {}
    count = 0
    with open(vocab_file) as f:
        for line in f.readlines():
            dict[count] = line.strip()
            count +=1
    return dict
    
def word_piece_to_word(tmp_ids):
    reverse_dict = get_ids_to_words_dict()
    list_of_tokens = []
    for ID in tmp_ids[1:]:
        list_of_tokens.append(reverse_dict[ID])
        if ID == 102:
            break
    list_of_words = []
    prev_token = ''
    for token in list_of_tokens:
        tmp_token = token
        if token[0] == '#':
            tmp_token = tmp_token[2:]
            prev_token=prev_token+tmp_token
        else:
            if prev_token:
                list_of_words.append(prev_token)
            prev_token = tmp_token
    return list_of_words
            
            
def word_piece_to_word_with_score(tmp_ids,word_scores):
    reverse_dict = get_ids_to_words_dict()
    word_scores = word_scores[1:]
    list_of_tokens = []
    for ID in tmp_ids[1:]:
        list_of_tokens.append(reverse_dict[ID])
        if ID == 102:
            break
    list_of_words = []
    new_word_scores = []
    prev_token = ''
    tmp_score = []
    for i in range(len(list_of_tokens)):
        tmp_token = list_of_tokens[i]
        if tmp_token[0] == '#':
            tmp_token = tmp_token[2:]
            prev_token=prev_token+tmp_token
            tmp_score.append(word_scores[i])
        else:
            if prev_token:
                score_of_word = max(tmp_score)
                new_word_scores.append(score_of_word)
                list_of_words.append(prev_token)
            prev_token = tmp_token
            tmp_score = [word_scores[i]]
            #pdb.set_trace()
    return list_of_words,new_word_scores
    
def One_layer_back_trace(att_heads): # need a tensor as input
    seq_len = att_heads.shape[2]
    num_head = att_heads.shape[1]
    layer_num = len(att_heads)
    last_layer_portion = np.diag([1] * seq_len)
    one_layer_att_heads = att_heads[0]
    sum_through_heads = torch.sum(one_layer_att_heads,0)
    sum_through_heads = sum_through_heads.cpu().detach().numpy()
    #sum_through_heads = np.where(sum_through_heads > 0.001, sum_through_heads, sum_through_heads -1000)
    prop_through_heads = sum_through_heads/num_head #softmax(sum_through_heads,axis=1)
    curr_layer_portion = np.matmul(prop_through_heads, last_layer_portion)
    return curr_layer_portion
    
def sent_score_to_csv(doc,sent_att_output,des_file = 'sent_scores.csv'):
    sent_weight = One_layer_back_trace(sent_att_output[0])
    data_to_df = []
    for i in range(len(doc)):
        tmp_dict = {}
        tmp_dict['token'] = doc[i][0]
        tmp_dict['score'] = sent_weight[0,i+1]
        data_to_df.append(tmp_dict)
    df = pd.DataFrame(data_to_df) 
    df.to_csv(des_file,index = False)
    return df    
    
def reverse_dict(dict_src):
    dict_inversed = {}
    for key in dict_src.keys():
        dict_inversed[dict_src[key]] = key
    return dict_inversed
    
def show_word_new(vectorizer,model,num,for_pos = True):
    index_2_word_dict = reverse_dict(vectorizer.vocabulary_)
    # get positive numbers 
    if hasattr(model.coef_, 'todense'):
        coef = model.coef_.todense()
    else:
        coef = model.coef_
    if not for_pos:
        coef = -coef
    coef = coef - np.mean(coef)
    pos_indices = np.where(coef>0)[1]
    pos_word = coef[0,pos_indices]
    descending_sort_pos_words = np.argsort(-pos_word)
    important_word_index = []
    important_word = []
    count =0
    index_count = 0
    while count < num:
        try:
            tmp_index = pos_indices[descending_sort_pos_words[0,index_count]]
        except:
            tmp_index = pos_indices[descending_sort_pos_words[index_count]]
        important_word_index.append(tmp_index)
        count+=1
        index_count += 1
    print(coef[0,tmp_index])
    for index in important_word_index:
        important_word.append(index_2_word_dict[index])
    return important_word
    
def vocab_match(vocab_file,keywords):
    vocab_dict = {}
    lines = open(vocab_file).readlines()
    for line in lines:
        vocab_dict[line.strip()]
    
def define_model(hpara_dict_path,trained_word_model,trained_sent_model):
    hpara_dict = json.load(open(hpara_dict_path))
    hpara1 = hpara()
    hpara1.__dict__.update(hpara_dict)
    
    #load bert config
    if hpara1.use_SSI_Bert:
        pretrain_model_dir = './BERT/bert/infection_bert_fix_gast'
    else:
        pretrain_model_dir = './BERT/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000'
    vocab_file = os.path.join(pretrain_model_dir,'vocab.txt')
    tokenizer = _load_tf_tokenizer(vocab_file = vocab_file)
    bert_config_file = os.path.join(pretrain_model_dir,'bert_config.json')
    
    #word model
    config = modeling_bert.BertConfig.from_json_file(bert_config_file)
    config.num_hidden_layers = hpara1.word_layers
    config.output_attentions = True
    model_word = modeling_bert.BertModel(config)
    model_word.load_state_dict(torch.load(trained_word_model))
    cls_weight = model_word.state_dict()['embeddings.word_embeddings.weight'][101]
    
    #sent model
    config_doc = modeling_bert.BertConfig.from_json_file(bert_config_file)
    config_doc.num_hidden_layers = hpara1.sent_layers
    config_doc.output_attentions = True
    config_doc.num_attention_heads = hpara1.head_num
    use_position_embedding = hpara1.use_position_embedding
    model_sent = modeling_bert.BertModel_no_embedding(config_doc,cls_weight,use_position_embedding=use_position_embedding)
    model_sent.load_state_dict(torch.load(trained_sent_model))
    
    # mount to GPGPU
    model_word = model_word.cuda()
    model_sent = model_sent.cuda()
    return hpara1,tokenizer,model_word, model_sent

def columns_count(df,key):
    print(df[key].isna().sum() / len(df))
    if len(df[key].unique())<5:
        df[key].value_counts(normalize=True).plot(kind='bar',figsize=(6, 8));
    else:
        df[key].value_counts(normalize=True).plot(kind='barh',figsize=(10, 20));
abb_list = ['y\.o\.M ', 'P\.M\.', 'M\.D ', 'E\.W\.', 'y\.o\.f ', 'e\.R ', 'h\.o', 'b\.s\.', 'M\.C\.', 'r\.n ', 'u\.o ', 'H\.C\.', 'N\.P\.O\.', 'o\.u\.', 'y\.o\.f\.', 'p\.m\.', 't\.E ', 'e\.A ', 'e\.g\.', 'n\.K ', 'U\.S\.', 'M\.F\.', 'y\.o\.m\.', 'q\.s\.', 'Y\.O ', 'a\.k\.a\.', 's\.o\.b\.', 'O\.U\.', 'l\.s ', 't\.d\.', 'o\.r ', 'd\.A\.', 'O\.D\.', 'b\.m\.', 'R\.A\.', 't\.E', 'N\.Y\.', 'p\.r\.n\.', 'b\.i\.d\.', 'p\.m', 'm\.a\.e\.', 'q\.d\.', 'o\.k ', 's\.A ', 'B\.L\.', 'h\.o ', 'A\.D\.', 'F\.u ', 'v\.o\.', 'B\.I\.', 'S\.O\.B\.', 'e\.g ', 'l\.e', 'M\.D\.', 'i\.e\.', 'T\.V\.', 's\.c ', 'd\.r ', 'r\.n\.', 'r\.o ', 'N\.B\.', 'H\.O ', 'A\.C\.E ', 'N\.O\.S ', 'h\.d\.', 'P\.T\.', 'B\.S ', 'I\.E ', 'O\.S\.', 'e\.p\.', 'm\.r\.g\.', 'p\.o\.q\.d\.', 'm\.d\.', 'R\.N ', 't\.s ', 'd\.r\.', 'h\.l\.', 's\.T ', 'E\.R\.', 't\.E\.', 'I\.S\.', 'B\.W\.', 'd\.L ', 'u\.o\.', 'r\.A ', 'n\.p\.o\.', 'd\.t ', 'B\.A\.', 'S\.K ', 'a\.m\.', 'i\.e', 'H\.R ', 'c\.w ', 'q\.a\.m\.', 'D\.T ', 'i\.v\.', 'B\.P\.', 'q\.p\.m\.', 's\.R ', 'p\.r\.', 'P\.O\.', 'd\.o\.', 'C\.T\.', 'R\.K', 'T\.F\.', 'e\.a ', 'p\.s\.', 'Y\.O\.', 'E\.D ', 'M\.P\.H\.', 'P\.o\.', 'I\.R ', 'V\.A\.C\.', 'N\.E\.', 'f\.u ', 'e\.d\.', 't\.a ', 'd\.c ', 'y\.o M ', 'R\.T\.', 'I\.D\.', 's\.p ', 's\.l ', 'q\.s ', 'g\.i\.', 'q\.h\.s\.', 'p\.o\.q\.i\.d\.', 'p\.o\.b\.i\.d\.', 'M\.S\.', 'd\.A ', 'H\.o ', 'S\.T ', 'o\.t\.', 'a\.c\.', 'n\.p\.', 't\.A ', 'p\.e\.', 't\.l\.', 'o\.r\.', 'w\.o ', 'P\.R\.N\.', 'I\.V ', 's\.P\.', 'I\.J\.', 'I\.R\.', 'P\.S\.', 'n\.t ', 'c\.p\.', 't\.o ', 'U\.o\.', 'B\.I\.D', 'U\.O\.', 'p\.m ', 'N\.P\.', 'F\.R\.', 'n\.s\.', 'b\.i\.d', 'I\.S ', 'N\.T ', 'p\.o\.', 's\.A\.', 'i\.e ', 'N\.C\.', 'e\.d ', 'R\.R\.', 'y\.o\.', 'I\.E\.', 's\.D ', 'P\.A\.', 'b\.i\.d ', 'O\.R ', 'q\.h\.s\.p\.r\.n\.', 'P\.C\.', 's\.p\.', 's\.c\.b\.i\.d\.', 'S\.O\.', 'y\.A ', 'q\.i\.d\.p\.r\.n\.', 't\.A\.', 'O\.T\.', 's\.s\.', 'c\.i\.', 'C\.O\.', 'p\.o\.q\.h\.s\.p\.r\.n\.', 'B\.M\.', 'u\.s\.', 'T\.M\.', 'p\.o ', 'R\.D\.', 'N\.H\.', 'y\.o\.m ', 'n\.A ', 'i\.v ', 'A\.G\.', 't\.D ', 'p\.g\.', 'e\.t\.', 'B\.I\.D ', 'H\.R\.', 'd\.s ', 't\.r ', 'O\.K\.', 'D\.R\.', 'y\.o\.F ', 'r\.D ', 'N\.O\.', 'i\.s\.', 'n\.c\.', 'A\.m ', 'o\.k\.', 's\.r ', 'P\.S ', 'I\.P\.', 'E\.D\.', 'i\.d\.', 'G\.I\.', 'h\.s\.', 'O\.R\.', 'N\.P\.N\.', 'H\.D\.', 'P\.O\.D\.', 'n\.c', 'P\.M ', 'P\.O ', 's\.K ', 'S\.R ', 'h\.o\.', 'R\.I\.', 'p\.o\.q\.', 'e\.A\.', 'C\.I\.', 'r\.t\.', 'I\.D ', 'R\.O ', 'D\.M\.', 'G\.U\.', 'c\.o ', 'q\.o\.d\.', 'I\.O ', 'R\.O\.', 'E\.G\.', 'c\.t\.', 'c\.i ', 'h\.D\.', 'a\.m ', 'H\.L\.', 'e\.w\.', 'E\.C\.', 'T\.O\.', 's\.s ', 'S\.C ', 'U\.O ', 'p\.o\.t\.i\.d\.', 'e\.r ', 'g\.u\.', 'i\.u\.', 'R\.N\.s ', 'S\.W\.', 'd\.o', 'R\.N\.', 'D\.R ', 'C\.I ', 'M\.I\.', 'B\.P ', 'a\.m', 'o\.s\.', 'E\.L ', 'e\.L ', 'n\.c ', 'Q\.S\.', 'r\.o\.s\.', 'e\.d', 'H\.O\.', 'A\.M ', 'p\.t\.', 'd\.o ', 'y\.o ', 'E\.T ', 'I\.V\.', 'h\.p\.r\.n\.', 't\.K ', 'P\.E\.', 'U\.S ', 'T\.D\.', 'i\.v', 's\.L ', 'c\.o\.', 'S\.C\.', 'C\.O ', 'T\.O ', 'o\.C ', 'l\.A ', 'S\.L\.', 'p\.a\.', 'i\.v\.q\.', 't\.A', 'F\.B\.', 't\.i\.d\.', 'D\.L ', 'D\.I\.C\.', 'c\.l\.', 'C\.L\.', 'A\.M', 'o\.d\.', 'R\.O\.S\.', 'p\.o', 'S\.P ', 'J\.P\.', 'B\.S\.', 'd\.R ', 's\.o\.', 'p\.o\.q\.a\.m\.', 'C\.A\.R\.E ', 'i\.r\.', 'q\.i\.d\.', 'A\.m\.', 'y\.o F ', 'B\.O\.', 'B\.I\.D\.', 'D\.L\.', 's\.c\.', 'S\.O ', 'S\.S\.', 'v\.s\.', 'p\.o\.q\.h\.s\.', 'A\.M\.']

nopoint_abb_list = ['yoM ', 'PM', 'MD ', 'EW', 'yof ', 'eR ', 'ho', 'bs', 'MC', 'rn ', 'uo ', 'HC', 'NPO', 'ou', 'yof', 'pm', 'tE ', 'eA ', 'eg', 'nK ', 'US', 'MF', 'yom', 'qs', 'YO ', 'aka', 'sob', 'OU', 'ls ', 'td', 'or ', 'dA', 'OD', 'bm', 'RA', 'tE', 'NY', 'prn', 'bid', 'pm', 'mae', 'qd', 'ok ', 'sA ', 'BL', 'ho ', 'AD', 'Fu ', 'vo', 'BI', 'SOB', 'eg ', 'le', 'MD', 'ie', 'TV', 'sc ', 'dr ', 'rn', 'ro ', 'NB', 'HO ', 'ACE ', 'NOS ', 'hd', 'PT', 'BS ', 'IE ', 'OS', 'ep', 'mrg', 'poqd', 'md', 'RN ', 'ts ', 'dr', 'hl', 'sT ', 'ER', 'tE', 'IS', 'BW', 'dL ', 'uo', 'rA ', 'npo', 'dt ', 'BA', 'SK ', 'am', 'ie', 'HR ', 'cw ', 'qam', 'DT ', 'iv', 'BP', 'qpm', 'sR ', 'pr', 'PO', 'do', 'CT', 'RK', 'TF', 'ea ', 'ps', 'YO', 'ED ', 'MPH', 'Po', 'IR ', 'VAC', 'NE', 'fu ', 'ed', 'ta ', 'dc ', 'yo M ', 'RT', 'ID', 'sp ', 'sl ', 'qs ', 'gi', 'qhs', 'poqid', 'pobid', 'MS', 'dA ', 'Ho ', 'ST ', 'ot', 'ac', 'np', 'tA ', 'pe', 'tl', 'or', 'wo ', 'PRN', 'IV ', 'sP', 'IJ', 'IR', 'PS', 'nt ', 'cp', 'to ', 'Uo', 'BID', 'UO', 'pm ', 'NP', 'FR', 'ns', 'bid', 'IS ', 'NT ', 'po', 'sA', 'ie ', 'NC', 'ed ', 'RR', 'yo', 'IE', 'sD ', 'PA', 'bid ', 'OR ', 'qhsprn', 'PC', 'sp', 'scbid', 'SO', 'yA ', 'qidprn', 'tA', 'OT', 'ss', 'ci', 'CO', 'poqhsprn', 'BM', 'us', 'TM', 'po ', 'RD', 'NH', 'yom ', 'nA ', 'iv ', 'AG', 'tD ', 'pg', 'et', 'BID ', 'HR', 'ds ', 'tr ', 'OK', 'DR', 'yoF ', 'rD ', 'NO', 'is', 'nc', 'Am ', 'ok', 'sr ', 'PS ', 'IP', 'ED', 'id', 'GI', 'hs', 'OR', 'NPN', 'HD', 'POD', 'nc', 'PM ', 'PO ', 'sK ', 'SR ', 'ho', 'RI', 'poq', 'eA', 'CI', 'rt', 'ID ', 'RO ', 'DM', 'GU', 'co ', 'qod', 'IO ', 'RO', 'EG', 'ct', 'ci ', 'hD', 'am ', 'HL', 'ew', 'EC', 'TO', 'ss ', 'SC ', 'UO ', 'potid', 'er ', 'gu', 'iu', 'RNs ', 'SW', 'do', 'RN', 'DR ', 'CI ', 'MI', 'BP ', 'am', 'os', 'EL ', 'eL ', 'nc ', 'QS', 'ros', 'ed', 'HO', 'AM ', 'pt', 'do ', 'yo ', 'ET ', 'IV', 'hprn', 'tK ', 'PE', 'US ', 'TD', 'iv', 'sL ', 'co', 'SC', 'CO ', 'TO ', 'oC ', 'lA ', 'SL', 'pa', 'ivq', 'tA', 'FB', 'tid', 'DL ', 'DIC', 'cl', 'CL', 'AM', 'od', 'ROS', 'po', 'SP ', 'JP', 'BS', 'dR ', 'so', 'poqam', 'CARE ', 'ir', 'qid', 'Am', 'yo F ', 'BO', 'BID', 'DL', 'sc', 'SO ', 'SS', 'vs', 'poqhs', 'AM']

def _batch_clean(note_text,num,start):
    start = int(start)
    local_seg = note_text[start:min(len(note_text),start+num)]
    local_clean_list = []
    for text in tqdm(local_seg):
        text = re.sub(r'\s*\[.*?\]\s*',' ',text)
        if re.search(r'[a-zA-Z]\.[a-zA-Z]',text):
            for i in range(len(abb_list)):
                text = re.sub(abb_list[i],nopoint_abb_list[i],text)
        text = re.sub(r'\n',' ',text)
        text = re.sub(r' :',':',text)
        #text = re.sub(r':\w',': ',text)
        text = re.sub(r'\s+',' ',text)
        local_clean_list.append(text)
    return local_clean_list
    
def clean_Abb(p_num,note_text):
    num = int(len(note_text)/p_num)
    p = Pool(processes = p_num)
    whole_clean_list = []
    func = partial(_batch_clean,note_text,num)
    clean_lists = p.map(func,range(0,len(note_text),num))
    p.close()  
    for lll in clean_lists:
        whole_clean_list.extend(lll)
    return whole_clean_list
    
def merge_20_cleanAbb(src_path):
    merge_to = 20
    src_text = pickle.load(open(src_path,'rb'))
    merged_text = []
    for text in src_text:
        tmp_doc = []
        buffer = ''
        cur_len = 0
        for line in text:
            line = re.sub(r'\n',' ',line)
            buffer = buffer + line+' ' 
            cur_len = cur_len + len(line.split())
            if cur_len>=merge_to:
                if re.search(r'[a-zA-Z]\.[a-zA-Z]',buffer):
                    for i in range(len(abb_list)):
                        buffer = re.sub(abb_list[i],nopoint_abb_list[i],buffer)
                tmp_doc.append(buffer)
                buffer = ''
                cur_len = 0
        if cur_len>0:
            tmp_doc.append(buffer)
        merged_text.append(tmp_doc)
    filename = src_path[:-4]
    des_path = filename[:-1] + '_merged.pkl'
    pickle.dump(merged_text,open(des_path,'wb'))
    return merged_text