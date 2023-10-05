#! /home/lucas/miniconda3/envs fresh

import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
from datetime import datetime
from torch.masked import masked_tensor, as_masked_tensor
#import warnings

class Transformer(nn.Module):
    def __init__(self, 
                 input_feature_size=768, 
                 input_sequence_length = 511, 
                 nhead=8, 
                 numlayers=3, 
                 dropout_in=0.4, 
                 dropout_transformer=0.3, 
                 dropout_out=0.2, 
                 condense_channels=8, 
                 kernel_size=25):
        super().__init__()
        self.linear_proj = nn.Sequential(
            nn.Dropout(p=dropout_in),
            nn.Linear(input_feature_size, input_feature_size // 2),
            nn.InstanceNorm1d(511),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_out),
            nn.Linear(input_feature_size // 2, input_feature_size // 4),
            nn.InstanceNorm1d(511),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_out),
            nn.Linear(input_feature_size // 4, input_feature_size // 4),
        )

        lstm_input_feature_size = input_feature_size // 4 + 144*2

        self.normalize = nn.InstanceNorm1d(lstm_input_feature_size)
        self.norm_embed = nn.InstanceNorm1d(input_feature_size)
        self.norm_attent = nn.InstanceNorm1d(144*2)

        self.encoderLayer = nn.TransformerEncoderLayer(
            d_model = lstm_input_feature_size,
            nhead = nhead,
            batch_first=True,
            dropout=dropout_transformer,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoderLayer,
            num_layers=numlayers
        )
        self.to_residue_small = nn.Sequential(
            nn.Linear(480, 480),
            #nn.InstanceNorm1d(480),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_out),
            nn.Linear(480, condense_channels),
        )

        self.no_conv_to_stability_small = nn.Sequential(
            nn.Flatten(),
            nn.Linear(condense_channels * 511, condense_channels * 511),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_out),
            nn.Linear(condense_channels * 511, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

        self.conv_to_stability_small = nn.Sequential(
            nn.Conv1d(in_channels=condense_channels, out_channels=10, kernel_size=kernel_size, bias=False), # L = 511 - 25 + 1 = 487
            nn.BatchNorm1d(num_features=10),
            nn.Flatten(),
            nn.Linear(10 * (511 + 1 - kernel_size), 10 * (511 + 1 - kernel_size)),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_out),
            nn.Linear(10 * (511 + 1 - kernel_size), 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, msa_query_embeddings, msa_attention_features):
        msa_query_embeddings = rearrange(msa_query_embeddings, 'N L F -> N F L')
        msa_query_embeddings = self.norm_embed(msa_query_embeddings)
        msa_query_embeddings = rearrange(msa_query_embeddings, 'N F L -> N L F')
        msa_query_embeddings = self.linear_proj(msa_query_embeddings)

        msa_attention_features = msa_attention_features[:,0,:,:]

        lstm_input = torch.cat([msa_query_embeddings, msa_attention_features], dim=2)
        lstm_input = rearrange(lstm_input, 'N L F -> N F L')
        lstm_input = self.normalize(lstm_input)
        lstm_input = rearrange(lstm_input, 'N F L -> N L F')
        lstm_output = self.encoder(lstm_input)
        label_output = self.to_residue_small(lstm_output)
        label_output = rearrange(label_output, 'N L F -> N F L')

        output = self.conv_to_stability_small(label_output)
        output = torch.reshape(output, (output.shape[0], 1))
        return output
    
class RNN(nn.Module):
    def __init__(self, 
                 input_feature_size=768,
                 rnn_hidden=1024,
                 mlp2_hidden=512,
                 lstm_layers=3,
                 dropout_in=0.4, 
                 dropout_lstm=0.3, 
                 dropout_out=0.2):
        super().__init__()
        self.linear_proj = nn.Sequential(
            nn.Dropout(p=dropout_in),
            nn.Linear(input_feature_size, input_feature_size // 2),
            nn.InstanceNorm1d(511, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_out),
            nn.Linear(input_feature_size // 2, input_feature_size // 4),
            nn.InstanceNorm1d(511, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_out),
            nn.Linear(input_feature_size // 4, input_feature_size // 4),
        )

        lstm_input_feature_size = input_feature_size // 4 + 144 *2
        self.normalize = nn.InstanceNorm1d(lstm_input_feature_size)
        self.norm_embed = nn.InstanceNorm1d(input_feature_size)
        self.norm_attent = nn.InstanceNorm1d(144*2)

        self.rnn_lstm = nn.GRU(input_size=lstm_input_feature_size, 
                                hidden_size=rnn_hidden // 2, #divide by 2 since by-directional will double the  number of hidden
                                num_layers=lstm_layers, 
                                dropout=dropout_lstm, 
                                bidirectional=True, 
                                batch_first=True)

        self.mlp2 = nn.Sequential(
            nn.Linear(rnn_hidden, mlp2_hidden),
            nn.LeakyReLU(),
            nn.Linear(mlp2_hidden, mlp2_hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(mlp2_hidden //2 , 1)
        )

    def forward(self, msa_query_embeddings, msa_attention_features):
        msa_query_embeddings = rearrange(msa_query_embeddings, 'N L F -> N F L')
        msa_query_embeddings = self.norm_embed(msa_query_embeddings)
        msa_query_embeddings = rearrange(msa_query_embeddings, 'N F L -> N L F')
        msa_query_embeddings = self.linear_proj(msa_query_embeddings)
        lstm_input = torch.cat([msa_query_embeddings, msa_attention_features], dim=2)
        lstm_input = rearrange(lstm_input, 'N L F -> N F L')
        lstm_input = self.normalize(lstm_input)
        lstm_input = rearrange(lstm_input, 'N F L -> N L F')


        out, hidden = self.rnn_lstm(lstm_input)
        forward_out = hidden[2, :, :] # should be lstm_layer-1
        reverse_out = hidden[-1,:,:]
        out = torch.cat([forward_out, reverse_out], dim=1)
        out = self.mlp2(out)
        return out

class RNN_light(nn.Module):
    def __init__(self, 
                 input_feature_size=768,
                 rnn_hidden=512,
                 mlp2_hidden=128,
                 lstm_layers=2,
                 dropout_in=0.4, 
                 dropout_lstm=0.3, 
                 dropout_out=0.2):
        super().__init__()
        self.linear_proj = nn.Sequential(
            nn.Dropout(p=dropout_in),
            nn.Linear(input_feature_size, input_feature_size // 2),
            nn.InstanceNorm1d(511, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_out),
            nn.Linear(input_feature_size // 2, input_feature_size // 4),
            nn.InstanceNorm1d(511, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_out),
            nn.Linear(input_feature_size // 4, input_feature_size // 4),
        )

        lstm_input_feature_size = input_feature_size // 4 + 144 *2
        self.normalize = nn.InstanceNorm1d(lstm_input_feature_size)
        self.norm_embed = nn.InstanceNorm1d(input_feature_size)
        self.norm_attent = nn.InstanceNorm1d(144*2)

        self.rnn_lstm = nn.GRU(input_size=lstm_input_feature_size, 
                                hidden_size=rnn_hidden // 2, #divide by 2 since by-directional will double the  number of hidden
                                num_layers=lstm_layers, 
                                dropout=dropout_lstm, 
                                bidirectional=True, 
                                batch_first=True)

        self.mlp2 = nn.Sequential(
            nn.Linear(rnn_hidden, mlp2_hidden),
            nn.LeakyReLU(),
            nn.Linear(mlp2_hidden, mlp2_hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(mlp2_hidden //2 , 1)
        )

    def forward(self, msa_query_embeddings, msa_attention_features):
        msa_query_embeddings = torch.permute(msa_query_embeddings, (0, 2, 1))
        msa_query_embeddings = self.norm_embed(msa_query_embeddings)
        msa_query_embeddings = torch.permute(msa_query_embeddings, (0, 2, 1))
        msa_query_embeddings = self.linear_proj(msa_query_embeddings)
        lstm_input = torch.cat([msa_query_embeddings, msa_attention_features], dim=2)
        lstm_input = torch.permute(lstm_input, (0, 2, 1))
        lstm_input = self.normalize(lstm_input)
        lstm_input = torch.permute(lstm_input, (0, 2, 1))


        out, hidden = self.rnn_lstm(lstm_input)
        forward_out = hidden[2, :, :] # should be lstm_layer-1
        reverse_out = hidden[-1,:,:]
        out = torch.cat([forward_out, reverse_out], dim=1)
        out = self.mlp2(out)
        return out

class RNN_Dataset(Dataset):
    """ Keeps track of the embeddings, attentions and the label value.

    :param selection: The UniProt_IDs that should be included in the dataset.
    :param device: which GPU or CPU to load the tensors to.
    :param database_file: Path to a csv file containing the columns 'UniProt_ID', 'MUTATION' and label_name.
    :param label_name: The name of the column containing the property to predict, eg "Tm" or "dG".
    :param embed_dir: The path to the directory of the embeddings. The file names have the following structure UniProt_ID_MUTATION_.pt
    :param attent_dir: The path to the directory of the attentions. The file names have the following structure UniProt_ID_MUTATION_.pt
    :param prefered_sample: If the database has the columns 'species' and 'sample_type'. Entries beloning to prefered sample will be chosen over the other samples.
    :param only_wt: If True, only wildtype entries will be included.
    :param target_transform: Transform or scale the label value.
    :return: embedding tensor, attention tensor, label value formated as tensor, file name
    """


    def __init__(self, 
                    selection,
                    device, 
                    database_file,
                    label_name,
                    embed_dir = '/mnt/nasdata/lucas/data/query_embed/', 
                    attent_dir = '/mnt/nasdata/lucas/data/condensed_row_attent/',
                    prefered_sample = None,
                    only_wt = False, 
                    transform=None, 
                    target_transform=None):
        
        database = pd.read_csv(database_file)
        database['name'] = database['UniProt_ID'] + '_' + database['MUTATION']
        self.device = device

        if prefered_sample != None:
            species = database['species'].drop_duplicates().tolist()
            for s in species:
                samples = database[database['species'] == s]['sample_type'].drop_duplicates().tolist()
                if len(samples) > 1:
                    database = database.drop(database[(database['species'] == s) & (database['sample_type'] != prefered_sample)].index.tolist())
                    
        if only_wt:
            database = database[database['MUTATION'] == 'WT']
        database['keep'] = False
        for id in selection:
            match = database['name'] == id
            database['keep'] += match 
        self.database = database[database['keep'] == True]
        self.label_name = label_name
        self.embed_dir = embed_dir
        self.attent_dir = attent_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.database.shape[0]

    def __getitem__(self, idx):
        #Load and padd attention tensor
        attent_path = self.attent_dir +  self.database['name'].iloc[idx] + '_.pt'
        attentions = torch.load(attent_path, map_location=self.device)
        attentions = attentions[0,:,:]

        #Load and padd embedding tensor
        embed_path = self.embed_dir + self.database['name'].iloc[idx] + '_.pt'
        embedding = torch.load(embed_path, map_location=self.device)

        #Cut off the padding of attention and embedding
        seqlen =  int(self.database['seqlen'].iloc[idx])
        attentions = attentions[:seqlen+1,:]
        embedding = embedding[:+seqlen+1,:]

        #Label value
        label = self.database[self.label_name].iloc[idx]
        label = torch.tensor(label)

        if self.transform:
            attentions = self.transform(attentions)
            embeddings = self.transform(embeddings)
        if self.target_transform:
            label = self.target_transform(label)
        return embedding, attentions, label, self.database['name'].iloc[idx]

def collate_batch(batch: Dataset):
    """ Enables mini-batch padding, meaning that the entries are padded until they match the length of the longest seuqnce.
    """
    embedding_list, attentions_list, label_list, names_list, max_length = [], [], [], [], -1

    for _embedding, _attentions, _label, _namnes in batch:
        embedding_list.append(_embedding)
        attentions_list.append(_attentions)
        label_list.append(_label)
        names_list.append(_namnes)
        if _attentions.shape[0] > max_length: #seqlen
            max_length = _attentions.shape[0]    

    padded_embedding_list = nn.utils.rnn.pad_sequence(embedding_list, batch_first=True)
    padded_attention_list = nn.utils.rnn.pad_sequence(attentions_list, batch_first=True)
    label_list = torch.tensor(label_list)
    label_list = rearrange(label_list, 'N -> N 1')

    return padded_embedding_list, padded_attention_list, label_list, names_list

class Only_embeddings(nn.Module):
    def __init__(self, input_feature_size=768, input_sequence_length = 511, dropout=0,):
        super().__init__()
        self.linear_proj = nn.Sequential(
            nn.Linear(input_feature_size, input_feature_size // 2),
            nn.InstanceNorm1d(input_feature_size // 2),
            nn.LeakyReLU(),
            nn.Linear(input_feature_size // 2, input_feature_size // 4),
            nn.InstanceNorm1d(input_feature_size // 2),
            nn.LeakyReLU(),
            nn.Linear(input_feature_size // 4, input_feature_size // 4),
        )

        transformer_input_feature_size = input_feature_size // 4

        self.normalize = nn.InstanceNorm1d(transformer_input_feature_size)
        self.norm_embed = nn.InstanceNorm1d(input_feature_size)
        #self.norm_attent = nn.InstanceNorm1d(144*2)

        self.encoderLayer = nn.TransformerEncoderLayer(
            d_model = transformer_input_feature_size,
            nhead = 8,
            batch_first=True,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoderLayer,
            num_layers=4
        )

        self.to_residue = nn.Sequential(
            nn.Linear(input_feature_size // 4, input_feature_size // 4),
            nn.InstanceNorm1d(input_feature_size // 4),
            nn.ReLU(),
            nn.Linear(input_feature_size // 4, input_feature_size // 8),
            nn.InstanceNorm1d(input_feature_size // 8),
            nn.ReLU(),
            nn.Linear(input_feature_size // 8, input_feature_size // 16),
            nn.InstanceNorm1d(input_feature_size // 16),
            nn.ReLU(),
            nn.Linear(input_feature_size // 16, 2),
        )

        self.conv_to_stability = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=10, kernel_size=25, bias=False), # L = 511 - 25 + 1 = 487
            nn.BatchNorm1d(num_features=10),
            nn.Flatten(),
            nn.Linear(10 * 487, 3 * 487),
            nn.InstanceNorm1d(3 * 487),
            nn.ReLU(),
            nn.Linear(3 * 487, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, msa_query_embeddings, _):
        msa_query_embeddings = self.linear_proj(msa_query_embeddings)
        lstm_input = self.normalize(msa_query_embeddings)
        lstm_output = self.encoder(lstm_input)
        label_output = self.to_residue(lstm_output)
        label_output = label_output.permute((0, 2, 1))
        output = self.conv_to_stability(label_output)
        output = torch.reshape(output, (output.shape[0], 1))

        return output

class S_pred_Dataset(Dataset):
    """ Keeps track of the the attentions and the normalized dG values.
    :param database: A csv file containing the columns 'UniProt_ID', 'MUTATION' and 'dG'.
    :param selection: The UniProt_IDs that belong to the fold.
    :param embed_dir: The directory of the embeddings. The file names have the following structure UniProt_ID_MUTATION.pt
    :param target_transform: Transform or scale the label value. By May 29th dividing by 30 was standard.
    :returns embedding tensor, attention tensor, label value formated as tensor, file name
    """
    def __init__(self, 
                    selection,
                    device, 
                    database_file = '/home/lucas/esmmsa/data/stability/ProTherm2.csv',
                    embed_dir = '/mnt/nasdata/lucas/data/query_embed/', 
                    attent_dir = '/mnt/nasdata/lucas/data/row_attent/',
                    only_wt = False, 
                    transform=None, 
                    target_transform=None):
        
        database = pd.read_csv(database_file)
        database['name'] = database['UniProt_ID'] + '_' + database['MUTATION']
        self.device = device
        if only_wt:
            database = database[database['MUTATION'] == 'WT']
        database['keep'] = False
        for id in selection:
            match = database['name'] == id
            database['keep'] += match 
        self.database = database[database['keep'] == True]
        self.embed_dir = embed_dir
        self.attent_dir = attent_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.database.shape[0]

    def __getitem__(self, idx):
        #Load and padd attention tensor
        attent_path = self.attent_dir +  self.database['name'].iloc[idx] + '_.pt'
        attentions = torch.load(attent_path, map_location=self.device)
        attentions = padd_row_attent(attentions, side='after')

        #Load and padd embedding tensor
        embed_path = self.embed_dir + self.database['name'].iloc[idx] + '_.pt'
        embedding = torch.load(embed_path, map_location=self.device)
        embedding = padd_embed(embedding, side='after')

        #Label value
        label = self.database['dG'].iloc[idx]
        label = torch.tensor(label)

        if self.transform:
            attentions = self.transform(attentions)
            embeddings = self.transform(embeddings)
        if self.target_transform:
            label = self.target_transform(label)
        return embedding, attentions, label, self.database['name'].iloc[idx]

class Transformer_Dataset_precondensed(Dataset):
    """ Keeps track of the embeddings, attentions and the label value.

    :param selection: The UniProt_IDs that should be included in the dataset.
    :param device: which GPU or CPU to load the tensors to.
    :param database_file: Path to a csv file containing the columns 'UniProt_ID', 'MUTATION' and label_name.
    :param label_name: The name of the column containing the property to predict, eg "Tm" or "dG".
    :param embed_dir: The path to the directory of the embeddings. The file names have the following structure UniProt_ID_MUTATION_.pt
    :param attent_dir: The path to the directory of the attentions. The file names have the following structure UniProt_ID_MUTATION_.pt
    :param prefered_sample: If the database has the columns 'species' and 'sample_type'. Entries beloning to prefered sample will be chosen over the other samples.
    :param only_wt: If True, only wildtype entries will be included.
    :param target_transform: Transform or scale the label value.
    :return: embedding tensor, attention tensor, label value formated as tensor, file name
    """
    def __init__(self, 
                selection: list,
                device, 
                database_file: str,
                label_name: str,
                embed_dir = '/mnt/nasdata/lucas/data/query_embed/', 
                attent_dir = '/mnt/nasdata/lucas/data/condensed_row_attent/',
                prefered_sample = None,
                only_wt = False, 
                transform=None, 
                target_transform=None):
        
        database = pd.read_csv(database_file)
        database['name'] = database['UniProt_ID'] + '_' + database['MUTATION']
        self.device = device

        if prefered_sample != None:
            species = database['species'].drop_duplicates().tolist()
            for s in species:
                samples = database[database['species'] == s]['sample_type'].drop_duplicates().tolist()
                if len(samples) > 1:
                    database = database.drop(database[(database['species'] == s) & (database['sample_type'] != prefered_sample)].index.tolist())
                    
        if only_wt:
            database = database[database['MUTATION'] == 'WT']
        database['keep'] = False
        for id in selection:
            match = database['name'] == id
            database['keep'] += match 
        self.database = database[database['keep'] == True]
        self.label_name = label_name
        self.embed_dir = embed_dir
        self.attent_dir = attent_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.database.shape[0]

    def __getitem__(self, idx):
        #Load and padd attention tensor
        attent_path = self.attent_dir +  self.database['name'].iloc[idx] + '_.pt'
        attentions = torch.load(attent_path, map_location=self.device)

        #Load and padd embedding tensor
        embed_path = self.embed_dir + self.database['name'].iloc[idx] + '_.pt'
        embedding = torch.load(embed_path, map_location=self.device)
        if embedding.shape[0] != 511:
            embedding = padd_embed(embedding, side='after')

        #Label value
        label = self.database[self.label_name].iloc[idx]
        label = torch.tensor(label)

        if self.transform:
            attentions = self.transform(attentions)
            embeddings = self.transform(embeddings)
        if self.target_transform:
            label = self.target_transform(label)
        return embedding, attentions, label, self.database['name'].iloc[idx]
    
class Only_Embeddings_Dataset(Dataset):
    #OBSOLETE. IGNORE  
    # HAVE TO CHANGE STRUCTURE OF TRAIN AND TEST LOOPS IN ORDER TO FUNCTION
    """ Keeps track of the the attentions and the normalized dG values.
    :param database: A csv file containing the columns 'UniProt_ID', 'MUTATION' and 'dG'.
    :param selection: The UniProt_IDs that belong to the fold.
    :param embed_dir: The directory of the embeddings. The file names have the following structure UniProt_ID_MUTATION.pt
    :param target_transform: Transform or scale the label value. By May 29th dividing by 30 was standard.
    :returns embedding tensor, attention tensor, label value formated as tensor, file name
    """
    def __init__(self, 
                    selection,
                    device, 
                    database_file = '/home/lucas/esmmsa/data/stability/ProTherm2.csv',
                    embed_dir = '/mnt/nasdata/lucas/data/query_embed/', 
                    only_wt = False, 
                    transform=None, 
                    target_transform=None):
        
        database = pd.read_csv(database_file)
        database['name'] = database['UniProt_ID'] + '_' + database['MUTATION']
        self.device = device
        if only_wt:
            database = database[database['MUTATION'] == 'WT']
        database['keep'] = False
        for id in selection:
            match = database['name'] == id
            database['keep'] += match 
        self.database = database[database['keep'] == True]
        self.embed_dir = embed_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.database.shape[0]

    def __getitem__(self, idx):
        #Dummy variable for attentions. Only used to make train and test loops happy
        attention = 'dummy'

        #Load and padd embedding tensor
        embed_path = self.embed_dir + self.database['name'].iloc[idx] + '_.pt'
        embedding = torch.load(embed_path, map_location=self.device)
        if embedding.shape[0] != 511:
            #print(f"ERROR: {self.database['name'].iloc[idx]}_.pt has the wrong embedding shape. Will perform an after-padd")
            embedding = padd_embed(embedding, side='after')

        #Label value
        label = self.database['Tm'].iloc[idx]
        label = torch.tensor(label)
        label = rearrange(label, 'N -> N 1')

        if self.transform:
            embeddings = self.transform(embeddings)
        if self.target_transform:
            label = self.target_transform(label)
        return embedding, attention, label, self.database['name'].iloc[idx]

def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, device, show=False) -> dict:
    """ Performs an epoch of training and updates the model parameters.
    
    input dataloader: the current dataset.
    input model: the model.
    input loss_fn: the loss funciton.
    input optimizer: the optimizer algorithm.
    input device: which GPU or CPU to run the model on.
    input show: if True, prints training progress.
    return: metrics relating to the performance of the training during the epoch."""

    torch.set_grad_enabled(True)
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    train_loss = 0
    PCC = 0
    MSE = 0
    RMSE = 0
    MAE = 0
    R2 = 0

    for batch, (emb, attent, y, _) in enumerate(dataloader):
        # Compute pred and loss
        pred = model(emb, attent)
        y = y.float().to(device)
        y = rearrange(y, 'N -> N 1')
        loss = loss_fn(pred, y)
        loss = loss.float()
        train_loss += loss.item()

        if pred.shape[0] > 1: #if a batch only contains 1 entry we get nan values for these metrics
            covar = torch.cov(torch.concat((pred, y), dim=1).T)
            pred_std = torch.std(pred)
            y_std = torch.std(y)
            temp = covar / (pred_std * y_std)
            PCC += temp[0,1]

            SSres = torch.sum(torch.pow(pred - y, 2))
            SStot = torch.sum(torch.pow(y - y.mean(), 2))
            R2 += 1 - (SSres/ SStot)

        MSE += torch.pow(pred - y, 2).mean()
        RMSE += torch.sqrt(torch.pow(pred - y, 2).mean())
        MAE += torch.abs(pred - y).mean()

        
        # Backpropagation
        for param in model.parameters(): #equivalent to optimizer.sero_grad() but supposedly more effective
            param.grad = None
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(emb)
            if show:
                now = datetime.now()
                current_time = now.strftime("%H:%M")
                print(f'\tloss: {loss:>7f} [{current:>5d}/{size:>5d}]\t{current_time}')
    train_loss /= num_batches
    PCC /= num_batches
    MSE /= num_batches
    RMSE /= num_batches
    MAE /= num_batches
    R2 /= num_batches

    PCC = PCC.item()
    MSE = MSE.item()
    RMSE = RMSE.item()
    MAE = MAE.item()
    R2 = R2.item()
    train = {'PCC': PCC, 'MSE':MSE, 'RMSE':RMSE, 'MAE':MAE, 'R2':R2}

    if show:
        print(f'\tAvg training loss: {train_loss:>7f}\tPCC: {PCC:>7f}')
    return train

def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn, device, mode, show=True) -> dict:
    """ Performs evaluation on a test set. Does NOT update model parameters. 
    
    input dataloader: the current dataset.
    input model: the model.
    input loss_fn: the loss funciton.
    input device: which GPU or CPU to run the model on.
    input show: if True, prints training progress.
    return: metrics relating to the performance of the dataset.
    """

    torch.set_grad_enabled(False)
    num_batches = len(dataloader)
    test_loss = 0
    PCC = 0
    MSE = 0
    RMSE = 0
    MAE = 0
    R2 = 0

    for emb, attent, y, _ in dataloader:
        pred = model(emb, attent)
        pred = pred.to(device)
        y = y.float().to(device)
        y = rearrange(y, 'N -> N 1')
        loss = loss_fn(pred, y)
        loss = loss.float()
        test_loss += loss.item()

        if pred.shape[0] > 1: #if a batch only contains 1 entry we get nan values for these metrics
            covar = torch.cov(torch.concat((pred, y), dim=1).T)
            pred_std = torch.std(pred)
            y_std = torch.std(y)
            temp = covar / (pred_std * y_std)
            PCC += temp[0,1]

            SSres = torch.sum(torch.pow(pred - y, 2))
            SStot = torch.sum(torch.pow(y - y.mean(), 2))
            R2 += 1 - (SSres/ SStot)


        MSE += torch.pow(pred - y, 2).mean()
        RMSE += torch.sqrt(torch.pow(pred - y, 2).mean())
        MAE += torch.abs(pred - y).mean()
    
    test_loss /= num_batches
    PCC /= num_batches
    MSE /= num_batches
    RMSE /= num_batches
    MAE /= num_batches
    R2 /= num_batches

    PCC = PCC.item()
    MSE = MSE.item()
    RMSE = RMSE.item()
    MAE = MAE.item()
    R2 = R2.item()
    test = {'PCC': PCC, 'MSE':MSE, 'RMSE':RMSE, 'MAE':MAE, 'R2':R2}

    if show:
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        if mode == 'val':
            print(f'\tAvg validation loss:  {test_loss:5f}\tPCC:  {PCC:5f}\t{current_time}\n')
        elif mode == 'test':
            print(f'\tAvg test loss:  {test_loss:5f}\tPCC:  {PCC:5f}\t{current_time}\n')
    return test

def transformer_inference(dataloader, model, device, show=True):
    """ Perform inference on entries.
    : return: a dictonary where the keys are UniProtID_MUTATION, the value is a dictionary of real_value and pred_value"""
    torch.set_grad_enabled(False)
    num_batches = len(dataloader)
    inference = {}

    for emb, attent, y, name in dataloader:
        pred = model(emb, attent)
        pred = pred.to(device)
        y = y.float().to(device)
        #y = rearrange(y, 'N -> N 1')
        
        for (_name, _y, _pred) in zip(name, y, pred):
            _y = _y.item()
            _pred = _pred.item()
            inference[_name] = {'real_value': _y, 
                                'pred_value': _pred}
        
            if show:
                print(f'\tInference of {_name}:\treal value {_y:.2f}\tPred value {_pred:.2f}')
    
    return inference

def RNN_inference(dataloader, model, device, show=True):
    """ Perform inference on entries.
    : return: a dictonary where the keys are UniProtID_MUTATION, the value is a dictionary of real_value and pred_value"""
    torch.set_grad_enabled(False)
    num_batches = len(dataloader)
    inference = {}

    for emb, attent, y, name in dataloader:
        pred = model(emb, attent)
        pred = pred.to(device)
        y = y.float().to(device)
        #y = rearrange(y, 'N -> N 1')
        
        for (_name, _y, _pred) in zip(name, y, pred):
            _y = _y.item()
            _pred = _pred.item()
            inference[_name] = {'real_value': _y, 
                                'pred_value': _pred}
        
            if show:
                print(f'\tInference of {_name}:\treal value {_y:.2f}\tPred value {_pred:.2f}')
    
    return inference

def padd_row_attent(attention_map, side):
    """ Padd attention maps to a size corresponding the map of an MSA of the size of MAX_MSA_COL_NUM x MAX_MSA_COL_NUM
        It is padded with 0. and the rows and columns are extended after the obtained values.
    
    : parameter attention_map: the representations to embed
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
        pad = [0, numpad, 0, numpad]

    for _ in range(attention_map.dim() - 2):
        pad.append(0)
        pad.append(0)
        

    padded_attention_map = torch.nn.functional.pad(input=attention_map,
                                                pad=pad,
                                                mode='constant',
                                                value=padd_value)
    return padded_attention_map

def padd_embed(embedding, side='both'):
    """ Padd embedding to a size corresponding the embedding of an MSA of the size of MAX_MSA_ROW_NUM x MAX_MSA_COL_NUM
    
    : parameter embedding: the representations to embed
    : return: the padded embedding, now at size [1, MAX_MSA_ROW_NUM, MAX_MSA_COL_NUM, 768]"""
    padd_value = 1
    MAX_MSA_ROW_NUM = 128  # 256
    MAX_MSA_COL_NUM = 512 - 1

    if side == 'both':
        if embedding.dim() == 2:
            length = embedding.shape[0]
            numpad = (MAX_MSA_COL_NUM - length) // 2
            evenout = 0
            if length + (2 * numpad) != MAX_MSA_COL_NUM:
                evenout = 1
            padded_embedding = torch.nn.functional.pad(input=embedding,
                                                        pad=(0, 0, numpad, numpad + evenout),
                                                        mode='constant',
                                                        value=padd_value)

        if embedding.dim() == 3:
            depth = embedding.shape[0]
            numpad = MAX_MSA_ROW_NUM - depth
            if numpad >= 0:
                padded_embedding = torch.nn.functional.pad(input=embedding,
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
            padded_embedding = torch.nn.functional.pad(input=padded_embedding,
                                                        pad=(0,0,numpad, numpad + evenout, 0, 0),
                                                        mode='constant',
                                                        value=padd_value)
            
    if side == 'after':
        if embedding.dim() == 2:
            length = embedding.shape[0]
            numpad = MAX_MSA_COL_NUM - length
            padded_embedding = torch.nn.functional.pad(input=embedding,
                                                        pad=(0, 0, 0, numpad),
                                                        mode='constant',
                                                        value=padd_value)

        if embedding.dim() == 3: #a bit unsure why this would be needed
            depth = embedding.shape[0]
            numpad = MAX_MSA_ROW_NUM - depth
            if numpad >= 0:
                padded_embedding = torch.nn.functional.pad(input=embedding,
                                                        pad=(0,0,0, 0, numpad, 0),
                                                        mode='constant',
                                                        value=padd_value)
            else:
                padded_embedding = embedding[:MAX_MSA_ROW_NUM,:]

            length = padded_embedding.shape[1]
            numpad = MAX_MSA_COL_NUM - length
            padded_embedding = torch.nn.functional.pad(input=padded_embedding,
                                                        pad=(0,0,0, numpad, 0, 0),
                                                        mode='constant',
                                                        value=padd_value)
    
    return padded_embedding

def cross_val(k, dir) -> list:
    """ Returns a list containing a touple of the devision between training and sampling of UniProtIDs. 
    Based on the clustering of TreeClust of the wildtypes MSA. Almost similar to K-fold cross validation.
    NOTE: No randomization before fold generation"""
    id_fold = pd.read_csv(dir)
    id_fold = id_fold[id_fold['Fold'] > -1]
    id_fold['name'] = id_fold['UniProt_ID'] + '_' + id_fold['MUTATION']
    folds = []
    for n in range(0,k):
        test_fold = id_fold.loc[id_fold['Fold'] == n]
        test = test_fold['name'].to_list()

        train_fold = id_fold.loc[id_fold['Fold'] != n]
        train = train_fold['name'].to_list()
        fold = (train, test)
        folds.append(fold)
    return folds

def folds_random(k, dir, label_name, test_species=None, set=None, only_wildtype=False) -> list:
    """ Returns a list of touples of the devision between training and sampling of UniProtIDs. 
    The separation into folds is random."""
    df = pd.read_csv(dir)
    df = df[df[label_name].notna()]
    if only_wildtype:
        df = df[df['MUTATION'] == 'WT']
    if set != None:
        df = df[df['set'] == set]
    if test_species != None:
        for s in test_species:
            df = df[df['species'] != s]
    df['name'] = df['UniProt_ID'] + '_' + df['MUTATION']
    #df = df[df['Fold'] > -1]
    length = df.shape[0]
    shuffled = np.arange(length)
    rng = np.random.default_rng()
    rng.shuffle(shuffled)
    folds = {}
    for i in range(k):
        folds[i] = {'training': [], 'testing' : []}

    for index in range(length):
        fold_num = index % k
        for n in range(0,k):
            if fold_num == n:
                folds[n]['testing'].append(df['name'].iloc[shuffled[index]])
            else:
                folds[n]['training'].append(df['name'].iloc[shuffled[index]])

    return_fold = []
    for n in range(0, k):
        fold = (folds[n]['training'], folds[n]['testing'])
        return_fold.append(fold)

    return return_fold

def cluster_random(k: int, db_path: str, label_name: str) -> list:
    """ 
    Returns a list of touples of the devision between training and validation.
    This function ensures that the entries of the same cluster end up in the 
    same fold, but randomizes which clusters end up in which fold.

    input k: the number of folds in k-fold cross validation.
    input db_path: the path of the csv-file containing the UniProtIDs and property described by the string label_name.
    input label_name: the name of the property the model should predict. Needs to be the column-name found in db_path.
    returns: a list of the separation between training and validation.
    """

    id_fold = pd.read_csv(db_path)
    id_fold['name'] = id_fold['UniProt_ID'] + '_' + id_fold['MUTATION']
    id_fold = id_fold[id_fold[label_name].notna()]
    leftout = id_fold[id_fold['cluster'] == -1]
    id_fold = id_fold[id_fold['cluster'] > -1]
    length = id_fold['cluster'].max()

    # Random generator
    shuffled = np.arange(length)
    rng = np.random.default_rng()
    rng.shuffle(shuffled)

    # Associates clusters randomly to folds.
    folds = {}
    for i in range(k):
        folds[i] = {'training': [], 'validation' : []}

    for index in range(length):
        fold_num = index % k
        for n in range(0,k):
            try:
                if fold_num == n:
                    folds[n]['validation'].append(id_fold.loc[id_fold['cluster'] == shuffled[index]]['name'].to_list())
                else:
                    folds[n]['training'].append(id_fold.loc[id_fold['cluster'] == shuffled[index]]['name'].to_list())
            except:
                print(f"Issue with cluster number {shuffled[index]}")

    # Associates the UniProtIDs of the clusters to the folds.
    return_fold = []
    for n in range(0, k):
        training = list()
        for sub_list in folds[n]['training']:
            training += sub_list
        validation = list()
        for sub_list in folds[n]['validation']:
            validation += sub_list

        fold = (training, validation)
        return_fold.append(fold)

    return return_fold

def folds_species(k, dir, test_species='-') -> list:
    """ Returns a list of touples of the devision between training and validation of UniProtIDs. The devision is solely based on the column 'species'.
    
    :param dir: the directory of the csv-file containing the UniProt_IDs, type of mutation, species and label value
    :param test_species: A species that should be exluced from the training or validation due to being used as test. 
    :return: a list of tuples where each tuple contains the division between training and validation by 'name' (UniProt_ID + '_' + 'MUTATION).
                [(training names fold 1, validation names fold 1), (training names fold 2, validation names fold 2), .., (training names fold n, validation names fold n)]"""


    id_fold = pd.read_csv(dir)
    if test_species != '-':
        id_fold = id_fold[id_fold['species'] != test_species] #I should include the information about which entries are kept blind
    id_fold['name'] = id_fold['UniProt_ID'] + '_' + id_fold['MUTATION']
    leftout = id_fold[id_fold['cluster'] == -1]
    id_fold = id_fold[id_fold['cluster'] > -1]
    folds = {}

    species = id_fold['species'].drop_duplicates().tolist()
    k = len(species)
    for n, s in enumerate(species):
        folds[n] = {'training': [], 'validation' : []}
        folds[n]['validation'].append(id_fold[id_fold['species'] == s]['name'].tolist())
        folds[n]['training'].append(id_fold[id_fold['species'] != s]['name'].tolist())

    return_fold = []
    for n in range(0, k):
        training = list()
        for sub_list in folds[n]['training']:
            training += sub_list
        validation = list()
        for sub_list in folds[n]['validation']:
            validation += sub_list

        fold = (training, validation)
        return_fold.append(fold)

    return return_fold

def ps_random(k, dir, mode='cv') -> list:
    """ Returns a list of touples of the devision between training and testing of UniProtIDs. 
    It can either be run in k-fold cross validation with mode='cv' or in a test mode without anty crossvalidation and with access to extra entries for testing with mode='blind' 
    This function ensures that the entries of the same cluster end up in the same fold, 
    but randomizes which clusters end up in which fold."""
    #Currently it is always the same entries within the cluster that are being used. I should implement a randomization, so that it is differeny every run.
    id_fold = pd.read_csv(dir)

    if mode == 'cv':
        id_fold = id_fold[id_fold['ps_group'] == 'train']
        id_fold['name'] = id_fold['UniProt_ID'] + '_' + id_fold['MUTATION']
        leftout = id_fold[id_fold['cluster'] == -1]
        id_fold = id_fold[id_fold['cluster'] > -1]
        length = id_fold['cluster'].max()
        shuffled = np.arange(length)
        rng = np.random.default_rng()
        rng.shuffle(shuffled)
        folds = {}
        for i in range(k):
            folds[i] = {'training': [], 'testing' : []}

        for index in range(length):
            fold_num = index % k
            for n in range(0,k):
                try:
                    if fold_num == n:
                        folds[n]['testing'].append(id_fold.loc[id_fold['cluster'] == shuffled[index]]['name'].to_list())
                    else:
                        folds[n]['training'].append(id_fold.loc[id_fold['cluster'] == shuffled[index]]['name'].to_list())
                except:
                    print(f"Issue with cluster number {shuffled[index]}")

        return_fold = []
        for n in range(0, k):
            training = list()
            for sub_list in folds[n]['training']:
                training += sub_list
            testing = list()
            for sub_list in folds[n]['testing']:
                testing += sub_list

            fold = (training, testing)
            return_fold.append(fold)

        return return_fold
    
    if mode =='blind':
        training = id_fold[id_fold['ps_group'] == 'train']
        testing = id_fold[id_fold['ps_group'] == 'test']

        training_list = training['name'].to_list()
        testing_list = training['name'].to_list()
        return [(training_list, testing_list)]

def normalize(up, typ) -> float:
    """
    Scales down the quantities so that they are approximately normalized.

    input up: the up-scaled value of the label.
    input typ: the label-name, must be either 'dG' or 'Tm'.
    return: the normalized or down-scaled value.
    """
    if typ == 'dG':
        mean = 6
        std = 5
    if typ == 'Tm':
        mean=331
        std=11
    return (up - mean) / std

def metricFromGuessingMean(values, mode):
    if mode == 'MAE':
        mean = values.mean()
        error = values - mean
        abs = np.abs(error)
        MAE = abs.mean().item()
        return MAE
    if mode == 'MSE':
        mean = values.mean()
        error = values - mean
        SE = np.power(error, 2)
        MSE = SE.mean().item()
        return MSE
    if mode == 'RMSE':
        MSE = metricFromGuessingMean(values, mode='MSE')
        RMSE = np.sqrt(MSE)
        return RMSE
