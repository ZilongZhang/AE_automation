import os
import csv
import pickle
import re
import pandas as pd
from tqdm import tqdm
import spacy
from multiprocessing import Pool,Process
import sys
import numpy as np
sys.path.append('../ClarityNLP/nlp')
sys.path.append('../ClarityNLP/')
sys.path.insert(1,'../tools')
from utils import concat_to_20
import Sentence_tokenizer_from_Clarity as cla_token
import json
from sklearn.model_selection import train_test_split
all_df = pd.read_csv('./result/pu_chart/pu_chart_nursing_3day.csv')
text_pair_dict = {}
label_pair_dict = {}
for _,row in all_df.iterrows():
    k = row.HADM_ID
    text_pair_dict[k] = row.clean_text
    label_pair_dict[k] = row['pu_label']

def _batch_cla_sent(start_index):
    tmp_sent_dict = {}
    sub_id_list = all_id[start_index:min(len(all_id),start_index+num)]
    for tmp_id in sub_id_list:
        list_sent = cla_token.parse_sentences_spacy(target_dict[tmp_id])
        tmp_sent_dict[tmp_id] = concat_to_20(list_sent)
    return tmp_sent_dict

p_num =20
target_dict = text_pair_dict
num = int(np.ceil(len(target_dict)/p_num))
all_id = list(target_dict.keys())
p = Pool(processes = p_num)
sent_dicts = p.map(_batch_cla_sent,range(0,len(target_dict),num))
p.close()
seg_dict = sent_dicts[0]
for ddd in sent_dicts:
    seg_dict.update(ddd)
print(len(seg_dict))
seg_list = []
for k in list(all_df.HADM_ID):
    seg_list.append(seg_dict[k])
all_df.seg_text = seg_list
import pdb; pdb.set_trace()
all_df.to_csv('./result/pu_chart/day3_data.csv')