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

#Load data
PATH = 'Protein_Stability_Masters_2023/Preprocessing/Meltome/'
df = pd.read_csv('cross-species.csv')

#Rename and reformat
df = df.dropna(subset=['meltPoint'])
df = df.rename(columns={'run_name':'species', 'Protein_ID':'UniProt_ID', 'meltPoint':'Tm'})

df = df.replace({'Arabidopsis thaliana seedling lysate': 'A.thaliana lysate',
                 'Thermus thermophilus HB27 lysate':'T.thermophilus lysate',
                 'Bacillus subtilis_168_lysate_R1':'B.subtilis lysate',
                 'Caenorhabditis elegans lysate':'C.elegans lysate',
                 'Danio rerio Zenodo lysate':'D.rerio lysate',
                 'Drosophila melanogaster SII lysate':'D.melanogaster lysate',
                 'Escherichia coli cells':'E.coli cells',
                 'Escherichia coli lysate':'E.coli lysate',
                 'Geobacillus stearothermophilus NCA26 lysate':'G.stearothermophilus lysate',
                 'Mus musculus BMDC lysate':'M.musculus lysate',
                 'Mus musculus liver lysate':'M.musculus lysate',
                 'Oleispira antarctica_RB-8_lysate_R1':'O.antartica lysate',
                 'Picrophilus torridus DSM9790 lysate':'P.torridus lysate',
                 'Saccharomyces cerevisiae lysate':'S.cerevisiae lysate',
                 'Thermus thermophilus HB27 cells':'T.thermophilus cells',
                 'Thermus thermophilus HB27 lysate':'T.thermophilus lysate',
                 'Homo sapiens Jurkat cells': 'H.sapiens cells',
                 'Homo sapiens K562 cells':'H.sapiens cells',
                 np.nan: '-'
                 })

#Calculate mean of stability value for entries of the same species and sample type
df = df.groupby(['species', 'UniProt_ID', 'gene_name'])
df = df.mean()
df = df.reset_index()

#% Alter so that the id is correct and not id_gene_name
df['UniProt_ID'] = df['UniProt_ID'].str.split('_')
for i in range(df.shape[0]):
    df.loc[:,'UniProt_ID'].iloc[i] = df['UniProt_ID'].iloc[i][0]

df = df.groupby(['species', 'UniProt_ID', 'gene_name'])
df = df.mean()
df = df.reset_index()

# Introducing a new column describing with sample type (instead of having the species column describing both species and sample type)
df['sample_type'] = '-'
lysates = [ 'A.thaliana lysate',
            'B.subtilis lysate',
            'C.elegans lysate',
            'D.melanogaster lysate',
            'D.rerio lysate', 
            'E.coli lysate',
            'G.stearothermophilus lysate',
            'M.musculus lysate',
            'O.antartica lysate',
            'P.torridus lysate',
            'S.cerevisiae lysate', 
            'T.thermophilus lysate']
cells = ['E.coli cells','T.thermophilus cells', 'H.sapiens cells']

for l in lysates:
    df.loc[df['species'] == l,'sample_type'] = 'lysate'
for c in cells:
    df.loc[df['species'] == c,'sample_type'] = 'cells'
df = df.replace({   'A.thaliana lysate': 'A.thaliana',
                    'B.subtilis lysate': 'B.subtilis',
                    'C.elegans lysate': 'C.elegans',
                    'D.melanogaster lysate': 'D.melanogaster',
                    'D.rerio lysate': 'D.rerio',
                    'E.coli cells': 'E.coli',
                    'E.coli lysate': 'E.coli',
                    'G.stearothermophilus lysate': 'G.stearothermophilus',
                    'H.sapiens cells': 'H.sapiens',
                    'M.musculus lysate': 'M.musculus',
                    'O.antartica lysate': 'O.antartica',
                    'P.torridus lysate': 'P.torridus',
                    'S.cerevisiae lysate': 'S.cerevisiae',
                    'T.thermophilus cells': 'T.thermophilus',
                    'T.thermophilus lysate': 'T.thermophilus'})

#%% Download sequences
#Function to make the gathering easier
class dumb:
    text = '-'
    def __init__():
        self.text = '-'

def get_url(url, **kvargs):
    response = requests.get(url, **kvargs);
    try:
        if not response.ok:
            print(response.text)
            response.raise_for_status()
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
progress = 0
WEBSITE_API = "https://rest.uniprot.org/uniprotkb/"
for id in IDs:
    if id != '-': 
        url = WEBSITE_API + id.strip() + '.fasta'
        FASTA = get_url(url).text
        df.loc[df['UniProt_ID'] == id, 'FASTA'] = FASTA
    elif id == '-': #For the Meltome database this if-loop will only be entered for human entries
        gene_names = df[df['UniProt_ID'] == '-']['gene_name'].drop_duplicates().tolist()
        for gene_name in gene_names:
            url = WEBSITE_API + f"search?query=gene:{gene_name}+AND+organism_id:9606&format=fasta"
            FASTA = get_url(url).text
            df.loc[(df['UniProt_ID'] == '-') & (df['gene_name'] == gene_name), 'FASTA'] = FASTA
    if progress % 1000 == 0:
        print(f"\tGathering {id}\n\tNumber {progress}\t{progress * 100//IDs.shape[0]:.2f}%")
    progress += 1
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
#%% Save-point before saving 
if not os.path.exists(os.path.join(os.getcwd(), 'processing')):
    os.mkdir('processing')
df.to_csv('processing/fasta_dump.csv')

#%% If the FASTAs already have been downloaded you can load here instead.
import pandas as pd
import numpy as np
df = pd.read_csv('processing/fasta_dump.csv')
df = df[df['FASTA'].notna()]
df = df[df['FASTA'] != '-']

fastas = df['FASTA'].str.split('>').tolist()
fastas = pd.DataFrame(fastas)
df['FASTA'] = '>' +fastas[1]


fastas = df['FASTA'].str.split('|').tolist()
uniprotids = pd.DataFrame(fastas)
df['UniProt_ID'] = uniprotids[1]

df['MUTATION'] = 'WT'
df['name'] = df['UniProt_ID'].astype('str') + '_' + df['MUTATION'].astype('str')
df['Tm'] = df['Tm'].astype('float32') + 273.15
df['temperature'] = df['temperature'].astype('float32') + 273.15
df = df[df['seqlen'] < 512]
df = df[df['seqlen'] > 40]

# Calculate the mean of entries from the same species, with same 'name' and sample type
duplicates = df[df.duplicated(subset=['species', 'name', 'sample_type'])]['name']
for d in duplicates:
    for s in ['lysate', 'cells']:
        mean = df.loc[(df['name'] == d) & (df['sample_type'] == s), 'Tm'].mean()
        df.loc[(df['name'] == d) & (df['sample_type'] == s), 'Tm'] = mean
df = df.drop_duplicates(subset=['species', 'name', 'sample_type'])

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

df = df.reset_index()
df = df.drop(columns=['index', 'Unnamed: 0', 'Unnamed: 0.1'])
df.to_csv('processed_Meltome.csv', index=False)

#%% Save all FASTAs as sepearate files in the directory seqdata, to enable MSA generation
if not os.path.exists(os.path.join(os.getcwd(), 'seqdata')):
    os.mkdir('seqdata')

index_list = ''
for index in range(df.shape[0]):
    file_name = df['name'].iloc[index] + '_.FASTA'
    index_list += file_name + '\n'
    path = 'seqdata/' + file_name
    with open(path, 'w') as file:
        file.write(df['FASTA'].iloc[index])

with open('seqdata/index.txt', 'w') as file:
    file.write(index_list)

#%% Save FASTAs for clustering
FASTA = str()
for fasta in df['FASTA']:
    FASTA += fasta


if not os.path.exists(os.path.join(os.getcwd(), 'cluster')):
    os.mkdir('cluster')

with open('cluster/all_seqs.FASTA', 'w') as file:
    file.write(FASTA)

# %% run CDhit didnt get this to work, so I did it manually instead.
# import subprocess
# os.chdir('/home/lucas/hh-suite/data/Meltome/cluster/')
# line = "cdhit -i all_seqs.FASTA -o all_seqs80 -c 0.8 -n 5 -d 0 -M 1600 -T 8"
# subprocess.run('mamba init \nmamba activate fresh\n' +line)
#os.system(line)

#%% Once clustering has been performed and saved the following can be run
CLUSTER_FILE_PATH = 'cluster/all_seqs80.clstr'
with open(CLUSTER_FILE_PATH, 'r') as file:
    txt = file.read()
ls = txt.split('>Cluster')
ls = ls[1:]
cl = pd.DataFrame(ls, columns=['name'])
cl = cl.reset_index()
cl['cluster'] = cl['index']

# The sequence with the * is the representative sequence. I have to relate all sequences to their representative sequences.
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

# Calculates how many entries are in each cluster
sum = 0
l = []
for k in range(cl.shape[0]):
    sum += len(cl['name'].iloc[k])
    l.append(len(cl['name'].iloc[k]))

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(l,'.')
plt.xlabel('Cluster number')
plt.ylabel('Number of sequences in cluster')
plt.figure(2)
plt.hist(l, bins=35)
plt.xlabel('Number ofsequences')
plt.ylabel('Frequency')

df['cluster'] = -1

for index in range(cl.shape[0]):
    names = cl['name'].iloc[index]
    for name in names:
        bol = df['name'] == name
        df.loc[bol, 'cluster'] = cl['cluster'].iloc[index]

#
df.to_csv('processing/Meltome_processed_clustered.csv', index=False)