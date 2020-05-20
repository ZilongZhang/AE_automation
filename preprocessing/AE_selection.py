import pandas as pd
import os
from collections import defaultdict

# Create a dictionary that map HADM_IDs to ICD codes
icd_lines =  open('../data/DIAGNOSES_ICD.csv').readlines()
adm_icd_dict = defaultdict(lambda:set())
for i in range(len(icd_lines)):
    split_line = icd_lines[i].split(',')
    adm = split_line[2]
    adm_icd_dict[adm].add(split_line[-1].strip()[1:-1])

json.dump(adm_icd_dict,open('./result/adm_icd_dict.json','w')) # Write this dictionary to file so we don't need to do it agian

# Find all HADM_IDs that have only one discharge summary described as a report.
# And create a dictionary that map those HADM_IDs to their DSs
note_df = pd.read_csv('../NOTEEVENTS.csv')
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


