import pandas as pd
import os
from collections import defaultdict
import imp
import sys
import re
import time
import json
import pickle
from tqdm import tqdm
import numpy as np
import utils
from multiprocessing import Pool
import pdb
from sklearn.model_selection import train_test_split

sys.path.append('/home/zilong.zhang1/AE_automation/ClarityNLP/nlp')
sys.path.append('/home/zilong.zhang1/AE_automation/ClarityNLP/')
import Sentence_tokenizer_from_Clarity as cla_token

# Create a dictionary that map HADM_IDs to ICD codes
icd_lines =  open('../data/DIAGNOSES_ICD.csv').readlines()
adm_icd_dict = defaultdict(lambda:set())
for i in range(len(icd_lines)):
    split_line = icd_lines[i].split(',')
    adm = split_line[2]
    adm_icd_dict[adm].add(split_line[-1].strip()[1:-1])
#json.dump(adm_icd_dict,open('./result/adm_icd_dict.json','w')) # Write this dictionary to file so we don't need to do it agian

# Find all HADM_IDs that have only one discharge summary described as a report.
# And create a dictionary that map those HADM_IDs to their DSs
note_df = pd.read_csv('../data/NOTEEVENTS.csv')
adm_count_dict = {}
ds_note = note_df.loc[note_df['CATEGORY'] == 'Discharge summary']
valid_ds_note = ds_note.loc[ds_note['DESCRIPTION'] == 'Report']
add_ds_note = ds_note.loc[ds_note['DESCRIPTION'] != 'Report']
raw_hadm_list =  list(valid_ds_note['HADM_ID'])
for hadm in raw_hadm_list: # Count how many valid DSs one HADM_ID has
    key = hadm
    value = raw_hadm_list.count(hadm)
    adm_count_dict[key] = value
    
valid_hadm_id = [] # The list of 
for key,value in adm_count_dict.items():
    if value == 1:
        valid_hadm_id.append(key)
pickle.dump(valid_hadm_id,open('./result/valid_hadm_id_list','wb')) #Save to disc

adm_text_dict = {}
for index,item in valid_ds_note.iterrows():
    if item['HADM_ID'] in valid_hadm_id:
        if item['HADM_ID'] in adm_text_dict:
            pdb.set_trace()
        adm_text_dict[item['HADM_ID']] = item['TEXT']
        
for index,item in add_ds_note.iterrows(): # merge addendum and report
    if item['HADM_ID'] in valid_hadm_id:
        adm_text_dict[item['HADM_ID']] = adm_text_dict[item['HADM_ID']] + str(item['TEXT'])

json.dump(adm_text_dict,open('./result/adm_text_dict.json','w'))



adm_seg_dict = json.load(open('./adm_seg_dict.json','r'))

AE_icds = ['99859','9093','99667','99666']
adm_with_AE = []
prev_key = 0
for key,value in adm_icd_dict.items():
    for icd in AE_icds:
        if icd in value:
            adm_with_AE.append(key)
            break
len(adm_with_AE)
pickle.dump(adm_with_AE,open('./valid_AE_adm','wb'))

AE_adm_seg_dict = {}
Neg_adm_seg_dict = {}

for key,value in adm_seg_dict.items():
    if str(key)[:-2] in adm_with_AE:
        AE_adm_seg_dict[key] = value
    else:
        Neg_adm_seg_dict[key] = value
print(len(AE_adm_seg_dict))
print(len(Neg_adm_seg_dict))

json.dump(AE_adm_seg_dict,open('./tmp/AE_adm_seg_dict','w'))
json.dump(Neg_adm_seg_dict,open('./tmp/Neg_adm_seg_dict','w'))

adm_list = []
total_seg_list = []
label_list = []
ratio = 2
count = 0
neg_num = len(AE_adm_seg_dict)*ratio
for key,value in AE_adm_seg_dict.items():
    adm_list.append(key)
    label_list.append(1)
    total_seg_list.append(value)

for key,value in Neg_adm_seg_dict.items():
    adm_list.append(key)
    label_list.append(0)
    total_seg_list.append(value)
    count +=1
    if count == neg_num:
        break
        
X_train, X_test, y_train, y_test,adm_train,adm_test = train_test_split(total_seg_list,label_list,adm_list,stratify=label_list,test_size=0.2)

data_dir = './result/ratio2/'
os.mkdir(data_dir)
with open(data_dir + 'text_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open(data_dir + 'text_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open(data_dir + 'label_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open(data_dir + 'label_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
with open(data_dir + 'hadm_train.pkl', 'wb') as f:
    pickle.dump(adm_train, f)
with open(data_dir + 'hadm_test.pkl', 'wb') as f:
    pickle.dump(adm_test, f)


