from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
from Bio import SeqIO
import string
import os
from scipy.spatial.distance import squareform, pdist, cdist
import numpy as np
from torch.nn import functional
os.chdir('/home/lucas/esmmsa')


# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

# Select sequences from the MSA to maximize the hamming distance
# Alternatively, can use hhfilter 
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa
    
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

def padd_embed(embedding, side):
    """ Padd embedding to a size corresponding the embedding of an MSA of the size of MAX_MSA_ROW_NUM x MAX_MSA_COL_NUM
    
    : parameter embedding: the representations to embed
    : parameter side: where the padding is concatinated, 
                      either 'both' to pad on both sides of the sequence, 
                      or 'after' pad only after the sequence.
    : return: the padded embedding, now at size [1, MAX_MSA_ROW_NUM, MAX_MSA_COL_NUM, 768]"""
    padd_value = 0
    MAX_MSA_ROW_NUM = 128  # 256
    MAX_MSA_COL_NUM = 512 - 1

    if side == 'both':
        if embedding.dim() == 2:
            length = embedding.shape[0]
            numpad = (MAX_MSA_COL_NUM - length) // 2
            evenout = 0
            if length + (2 * numpad) != MAX_MSA_COL_NUM:
                evenout = 1
            padded_embedding = functional.pad(input=embedding,
                                                        pad=(0, 0, numpad, numpad + evenout),
                                                        mode='constant',
                                                        value=padd_value)

        if embedding.dim() == 3:
            depth = embedding.shape[0]
            numpad = MAX_MSA_ROW_NUM - depth
            if numpad >= 0:
                padded_embedding = functional.pad(input=embedding,
                                                        pad=(0,0,0, 0, numpad, 0),
                                                        mode='constant',
                                                        value=padd_value)
            else:
                padded_embedding = embedding[:MAX_MSA_ROW_NUM,:]

            length = padded_embedding.shape[1]
            numpad = (MAX_MSA_COL_NUM - length) // 2
            evenout = 0
            if length + (2 * numpad) != MAX_MSA_COL_NUM:
                evenout = 1
            padded_embedding = functional.pad(input=padded_embedding,
                                                        pad=(0,0,numpad, numpad + evenout, 0, 0),
                                                        mode='constant',
                                                        value=padd_value)
            
    if side == 'after':
        if embedding.dim() == 2:
            length = embedding.shape[0]
            numpad = MAX_MSA_COL_NUM - length
            padded_embedding = functional.pad(input=embedding,
                                                        pad=(0, 0, 0, numpad),
                                                        mode='constant',
                                                        value=padd_value)

        if embedding.dim() == 3: #a bit unsure why this would be needed
            depth = embedding.shape[0]
            numpad = MAX_MSA_ROW_NUM - depth
            if numpad >= 0:
                padded_embedding = functional.pad(input=embedding,
                                                        pad=(0,0,0, 0, numpad, 0),
                                                        mode='constant',
                                                        value=padd_value)
            else:
                padded_embedding = embedding[:MAX_MSA_ROW_NUM,:]

            length = padded_embedding.shape[1]
            numpad = MAX_MSA_COL_NUM - length
            padded_embedding = functional.pad(input=padded_embedding,
                                                        pad=(0,0,0, numpad, 0, 0),
                                                        mode='constant',
                                                        value=padd_value)
    
    return padded_embedding

def padd_row_attent(attention_map, side):
    """ Padd attention maps to a size corresponding the map of an MSA of the size of MAX_MSA_COL_NUM x MAX_MSA_COL_NUM
        It is padded with 0. and the rows and columns are extended after the obtained values.
        The padded dimension is the last one in the tensor.
    
    : parameter attention_map: the attentions to embed
    : parameter side: where the padding is concatinated, 
                      either 'both' to pad on both sides of the sequence, 
                      or 'after' pad only after the sequence.
    : return: the padded attention_map, now at size [MAX_MSA_COL_NUM x MAX_MSA_COL_NUM]"""

    padd_value = 0
    MAX_MSA_COL_NUM = 512 - 1 
    length = attention_map.shape[-1]
    if side == 'both':
        numpad = (MAX_MSA_COL_NUM - length) // 2
        evenout = 0
        if length + (2 * numpad) != MAX_MSA_COL_NUM:
            evenout = 1
        pad = [numpad, numpad + evenout, 
            numpad, numpad + evenout]
    if side == 'after':
        numpad = MAX_MSA_COL_NUM - length
        pad = [0, numpad]
        
    padded_attention_map = functional.pad(input=attention_map,
                                                pad=pad,
                                                mode='constant',
                                                value=padd_value)
    return padded_attention_map