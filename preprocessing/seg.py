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
from utils import merge_cleanAbb
import Sentence_tokenizer_from_Clarity as cla_token
import json
from sklearn.model_selection import train_test_split
all_df = pd.read_csv('./result/pu_chart/pu_chart_nursing_3day.csv')
text_pair_dict = {}
label_pair_dict = {}
include_stage1 = False
suffix = 'have' if include_stage1 else 'no'
for _,row in all_df.iterrows():
    k = row.RECORD_ID
    text_pair_dict[k] = row.clean_text
    label_pair_dict[k] = row['pu_label']

def _batch_cla_sent(start_index):
    tmp_sent_dict = {}
    sub_id_list = all_id[start_index:min(len(all_id),start_index+num)]
    for tmp_id in sub_id_list:
        list_sent = cla_token.parse_sentences_spacy(target_dict[tmp_id])
        tmp_sent_dict[tmp_id] = merge_cleanAbb(list_sent)
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
#import pdb; pdb.set_trace()
all_seg_df = pd.DataFrame(columns = ['RECORD_ID','seg_text','pu_label','split'])
k_list = list(all_df.RECORD_ID)
for k in k_list:
    original_row = all_df[all_df.RECORD_ID == k].iloc[0]
    tmp_text = seg_dict[k]
    tmp_label = original_row['pu_label']
    tmp_split = original_row['split']
    df_length = len(all_seg_df)
    all_seg_df.loc[df_length] = [k,tmp_text,tmp_label,tmp_split]
#import pdb; pdb.set_trace()
json.dump(seg_dict,open('seg_dict.json','w'),indent=4,default=str)
all_seg_df.to_csv('./result/trial_data/day3_seg_{}1.csv'.format(suffix))