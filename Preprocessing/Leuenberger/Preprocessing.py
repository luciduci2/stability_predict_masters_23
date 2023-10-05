#%%
#!/usr/bin/env fresh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from scipy.optimize import fsolve
import requests, sys
import os
import random
save = False
#%%

PATH = '/home/lucas/Protein_Stability_Masters_2023/Preprocessing/Leuenberger/' #YOU WILL HAVE TO CHANGE
os.chdir(PATH)
species = ['E.coli',
           'H.sapiens',
           'S.cerevisiae',
           'T.thermophilus']

df = pd.DataFrame()
for name in species:
    temp = pd.read_csv(PATH + 'info_' + name + '.csv')
    temp['species'] = name
    df = pd.concat([df, temp[['tm', 'entry', 'species']]], axis=0)

df = df.reset_index(drop=True)
df.rename(columns={'tm': 'Tm', 'entry':'UniProt_ID'}, inplace=True)

#%% Download sequence (length is needed to estimate dG from Tm)
#Function to make the gathering easier
class dumb:
    text = '-'
    def __init__():
        self.text = '-'

def get_url(url, **kvargs):
    response = requests.get(url, **kvargs)
    try:
        if not response.ok:
            print(response.text)
            response.raise_for_status()
            #sys.exit()
    except requests.exceptions.HTTPError:
        print("ERROR: could not gather information from URL: equests.exceptions.HTTPError\n\tWill set FASTA to '-' and continue.")
        response = dumb
        
    return response

#Go through all of the UniProtIDs nonredundantly to gather the associated FASTA files
IDs = df['UniProt_ID']
IDs = IDs.drop_duplicates()
df['FASTA'] = '-'

print(f"Number of entries: {df.shape[0]}")
print(f"Number of unique proteins : {IDs.shape[0]}")
print(f"Start downloading FASTAS.")
#df = pd.read_csv('full_info_w_seq.csv')
#df = df.drop('Unnamed: 0', axis='columns')
progress = 0
WEBSITE_API = "https://rest.uniprot.org/uniprotkb/"
for id in IDs:
    url = WEBSITE_API + id.strip() + '.fasta'
    FASTA = get_url(url).text  
    df.loc[df['UniProt_ID'] == id, 'FASTA'] = FASTA
    if progress % 1000 == 0:
        print(f"\tGathering {id}\n\tNumber {progress}\t{progress * 100//IDs.shape[0]:.2f}%")
    progress += 1
df.to_csv('fasta_dump.csv')
print(f"Finished downloading sequenes. Sequences will be saved in fasta_dump.csv. Time to alter FASTAS")


#FASTA to sequence
df = df[df['FASTA'] != '-']
df['seq'] = df['FASTA'].str.split('\n')
for index in range(df.shape[0]):
    df['seq'].iloc[index] = df['seq'].iloc[index][1:]
df['seq'] = df['seq'].str.join('')

df['seqlen'] = -1
for index in range(df.shape[0]):
    df['seqlen'].iloc[index] = len(df['seq'].iloc[index])
df = df.drop('seq', axis='columns')

df.to_csv('full_info_w_seq.csv')

#%% If previous step already performed can load the file, so that the FASTAs don't have to be downloaded again.
df = pd.read_csv(PATH + 'full_info_w_seq.csv')
df = df[df['FASTA'] != '-']
df['MUTATION'] = 'WT'
df['name'] = df['UniProt_ID'].astype('str') + '_' + df['MUTATION'].astype('str')
df['Tm'] = df['Tm'].str.split('(')
for index in range(df.shape[0]):
    df['Tm'].iloc[index] = df['Tm'].iloc[index][0]
df['Tm'] = df['Tm'].astype('float32') + 273.15
df = df[df['seqlen'] < 512]
df = df[df['seqlen'] > 40]


listofduplicates = df['name'][df['name'].duplicated()].tolist()
for duplicate in listofduplicates:
    df.loc[df['name'] == duplicate, 'Tm'] = df[df['name'] == duplicate]['Tm'].mean()
df = df.drop_duplicates(subset='name')

#%% Make FASTA easier to handle for project
error = 0
for index in range(df.shape[0]):
    fasta = df['FASTA'].iloc[index]
    if pd.isnull(fasta):
        df['valid_mutation'].iloc[index] = False
        error += 1
    else:
        seq = fasta.split('\n')
        seq = ''.join(seq[1:])
        rows = len(seq) // 60 + 1
        r = 0
        s = ''
        for r in range(rows - 1):
            s += seq[r*60: (r+1)*60] + '\n'
        s += seq[(r+1)*60:] + '\n'

        df['FASTA'].iloc[index] = ">" + df['UniProt_ID'].iloc[index] + "_WT"+ "\n" + s

df.to_csv('Leuen_Pascal_therm.csv', index=False) #

#%% Save all FASTAs as sepearate files in the directory seqdata, to enable MSA generation
# You have to create the directory .../seqdata/ as well as .../cluster/

os.chdir(PATH)
index_list = ''
for index in range(df.shape[0]):
    file_name = df['name'].iloc[index] + '_.FASTA'
    index_list += file_name + '\n'
    with open(PATH + 'seqdata/' + file_name, 'w') as file:
        file.write(df['FASTA'].iloc[index])
if save:
    with open(PATH + 'seqdata/index.txt', 'w') as file:
        file.write(index_list)

#%% Save FASTAs for clustering
FASTA = str()
for fasta in df['FASTA']:
    FASTA += fasta

with open(PATH + 'cluster/all_seqs.FASTA', 'w') as file:
    file.write(FASTA)

# %% run CDhit to cluster the sequences with 80% sequence identity.
os.chdir(PATH + 'cluster/')
line = "cdhit -i all_seqs.FASTA -o all_seqs80 -c 0.8 -n 5 -d 0 -M 1600 -T 8"
os.system(line)

# %% Reads the clustering results and makes the results interpretable for the script.
CLUSTER_FILE_PATH = PATH + 'cluster/all_seqs80.clstr'
with open(CLUSTER_FILE_PATH, 'r') as file:
    txt = file.read()
ls = txt.split('>Cluster')
ls = ls[1:]
cl = pd.DataFrame(ls, columns=['name'])
cl = cl.reset_index()
cl['cluster'] = cl['index']

#%% The sequence with the * is the representative sequence. I have to relate all sequences to their representative sequences.
#repr2id = {}
clusters = []
repr_seq = ''
too_many_seqs = []
for index in range(cl.shape[0]):
    r = cl['name'].iloc[index]
    r = r.strip()
    r = r.split('\n')
    seqs = []
    for e in r:
        e = e.split(', ')[-1]
        if e[0] == '>':
            if e[-1] == '*': #representative sequence
                e = e.split('...')[0][1:]
                repr_seq = e
            else:
                e = e.split('...')[0][1:]

            seqs.append(e)
    cl['name'].iloc[index] = seqs

for index in range(cl.shape[0]):
    names = cl['name'].iloc[index]
    random.shuffle(names)
    if len(names) > 35:
        names = names[:35]
    cl['name'].iloc[index]= names

sum = 0
l = []
for k in range(cl.shape[0]):
    sum += len(df['name'].iloc[k])
    l.append(len(df['name'].iloc[k]))

plt.figure(1)
plt.plot(l,'.')
plt.xlabel('Cluster number')
plt.ylabel('Number of sequences in cluster')
plt.figure(2)
plt.hist(l, bins=35)
plt.xlabel('Number ofsequences')
plt.ylabel('Frequency')