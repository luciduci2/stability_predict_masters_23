#%%
#! /home/lucas/miniconda3/envs fresh
"""
Run this script after generating embeddings and attentions with the MSA Transformer. 
It compares the database with thermodynamic information to the directory containing the condensed row attentions 
and removes the entries that do not have attention generated.
You have to manually change:
    the paths of the database to check, 
    the path of the directionary holding all of the attentions,
    where the altered database should be saved.
"""

import os
import pandas as pd

df = pd.read_csv('ProThermDB/preprocess/ProTherm_preprocessed.csv')
attent_files = os.listdir('/mnt/nasdata/lucas/data/condensed_row_attent/')

names = df['UniProt_ID'] + '_' + df['MUTATION'] + '_.pt'

no_attention = []
for name in names:
    if name not in attent_files:
        no_attention.append(name)

to_remove = []
for e in no_attention:
    temp = e[:-4].split('_')
    temp = [temp[0], '_'.join(temp[1:])]
    to_remove.append(temp)

index_to_drop = []
for e in to_remove:
    temp = df[df['UniProt_ID'] == e[0]]
    temp = temp[temp['MUTATION'] == e[1]]
    index_to_drop.append(temp.index.to_list())

index_to_drop = [item for sublist in index_to_drop for item in sublist]
print(len(index_to_drop))
# %%
df = df.drop(index_to_drop, axis='index')
df.to_csv('ProThermDB/preprocess/ProTherm_embedded.csv',index=False)