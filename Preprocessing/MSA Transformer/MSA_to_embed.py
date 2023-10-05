#! /home/lucas/miniconda3/envs fresh
"""
!!  BEFORE RUNNING THIS SCRIPT YOU HAVE TO MANUALLY ALTER PATHS !!
MSA_DIR on row 36
EMBED_DIR on row 41
WANT TO CHANGE THE PATHS ON ROWS: 101 TO 115 !!

num_seq of 16 takes up 2,4 GB of GPU
num_seq of 128 takes up 22 GB of GPU

"""
import sys
from typing import List, Dict
import os
from datetime import datetime
import torch
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
from biotite.structure.io.pdbx import get_structure
import esm
from helperFunctions import read_msa, greedy_select, padd_embed, padd_row_attent
from einops import rearrange


now = datetime.now()
current_time = now.strftime("%H:%M")
print('Hello, starting up :)')
print(f"{current_time} STARTING MSA TRANSFORMER. LOADING MODEL")

torch.set_grad_enabled(False)
device = torch.device("cuda:0")
torch.cuda.set_device('cuda:0')

#%% Prepare data
MSA_DIR = '../Meltome/MSA/MSAs/'        #CHANGE TO THE DATABASE YOU WANT TO CREATE EMBEDDINGS AND ATTENTIONS FOR
fileNames = os.listdir(MSA_DIR)
#fileNames = fileNames[0:4]             #DE-COMMENT IF YOU ONLY WANT TO PROCESS 4 MSAs TO TEST
names = [name[:-4] for name in fileNames]

EMBED_DIR = '/mnt/nasdata/lucas/data/query_embed/' #path to directory with embeddings
CONDENSED_ROW_ATTENTION_DIR = '/mnt/nasdata/lucas/data/condensed_row_attent/'
alreadyDone = os.listdir(EMBED_DIR)
alreadyDone = [name[:-3] for name in alreadyDone]
toDo = []
for msa in names:
    if msa not in alreadyDone: #keeps track on if the protein already has embeddings
        toDo.append(msa)

msas = {
    name: read_msa(MSA_DIR+name+".a3m")
    for name in toDo
}

sequences = {
    name: msa[0] for name, msa in msas.items()
}

#%% Load the MSA-1b, get model, alphabet and converter
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S") 
model = model.eval().cuda().to(device)   # disables dropout for deterministic results
batch_converter = alphabet.get_batch_converter()

now = datetime.now()
current_time = now.strftime("%H:%M")
print(f"{current_time} Model loaded. Starting up generation of attention, embedding and contacts.")
print(f"{current_time} {str(len(toDo))} number of MSAs to perform inference on.")
results = {}
num_seqs=128 #the depth of the MSAs
for name, inputs in msas.items():
    query_embed_path = EMBED_DIR + name + '.pt'
    condensed_row_attent_path = CONDENSED_ROW_ATTENTION_DIR + name + '.pt'
    if os.path.exists(query_embed_path) & os.path.exists(condensed_row_attent_path):
            sys.stdout.write('INFO: query embedding and condensed row attention already found in output directory for ' + name + '\n')
    else:        
        try:
            input = greedy_select(inputs, num_seqs) #greedy_select maximizes the hamming distance
            batch_labels, batch_strs, batch_tokens = batch_converter(input)
            batch_tokens = batch_tokens.to(next(model.parameters()).device)
            batch_len = (batch_tokens != alphabet.padding_idx).sum(2)[0,0].item() #the length of the MSA

            # Extract per-residue representations
            with torch.no_grad():
                inference = model(batch_tokens, repr_layers=[12], return_contacts=True)

            #Embeddings
            query_embed = inference["representations"][12][0][0]
            query_embed = padd_embed(query_embed, side='after')
            torch.save(query_embed.clone(), query_embed_path)

            #Attentions
            row_attent = inference['row_attentions']
            row_attent = rearrange(row_attent, 'b l h i j -> b (l h) i j')
            row_attent = torch.cat((torch.mean(row_attent, dim=2), torch.mean(row_attent, dim=3)), dim=1)
            row_attent = padd_row_attent(row_attent, side='after')
            row_attent = row_attent.permute((0, 2, 1))
            torch.save(row_attent.clone(), condensed_row_attent_path) #shape: [1, MAX_SEQ_LENGTH, 288], 288 is derived from the NUM_HEADS * NUM_LAYERS * 2

            #Print status
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            print(current_time + ' SUCCESS: ' + name + ' has been processed.')
        except:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            print(current_time + ' ERROR: ' + name + ' could not properly processed.')