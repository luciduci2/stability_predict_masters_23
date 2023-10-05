#%%
#! /home/lucas/miniconda3/envs fresh

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os

df = pd.read_csv('/home/lucas/esmmsa/data/stability/Meltome2.csv')
blind = pd.read_csv('/home/lucas/esmmsa/data/handling/protstab2_blind.csv', index_col=0)
blind_list = blind['Protein ID'].tolist()

df['Protein ID'] = df['UniProt_ID'] + '_' + df['gene_name']
df['set'] = '-'
df['ProTstab2_pred'] = np.nan

for i in range(df.shape[0]):
    protein_id = df.loc[:,'Protein ID'].iloc[i]
    if  protein_id in blind_list:
        df.loc[:,'set'].iloc[i] = 'blind'
        df.loc[:,'ProTstab2_pred'].iloc[i] = blind.loc[blind['Protein ID'] == protein_id, 'Our_Predict'] + 273


species = df['species'].drop_duplicates().tolist()
for s in species:
    samples = df[df['species'] == s]['sample_type'].drop_duplicates().tolist()
    if len(samples) > 1:
        df = df.drop(df[(df['species'] == s) & (df['sample_type'] != 'lysate')].index.tolist())

df = df.reset_index(drop=True)

# %%

trainval_indices = np.array(df[df['set'] == '-'].index)

rng = np.random.default_rng()
rng.shuffle(trainval_indices)

num_val = len(trainval_indices) // 10
val_indices = trainval_indices[:num_val]
train_indices = trainval_indices[num_val:]

df.loc[val_indices,'set'] = 'validation'
df.loc[train_indices,'set'] = 'training'
#%%
df.to_csv('/home/lucas/esmmsa/data/stability/Meltome2.csv')
# %%
