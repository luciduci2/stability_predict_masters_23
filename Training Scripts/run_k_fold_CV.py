#%%
#! /home/lucas/miniconda3/envs fresh
""" This script performs k-fold cross validation of a model. It is easy to change the model and data to train and validate.
Find models in the file helper_functions.py
"""

import os
import torch
from torch import nn
#import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from datetime import datetime
import json
#import numpy as np
from helper_functions import *

# %% Parameters and variables you could and should alter
save_directory = 'trained_models_dump/'
device = torch.device('cuda:0')
learning_rate = 1e-4
weight_decay = 1e-3
folds = 5
epochs = 20
batch_size = 64
shuffle = True  #if the dataloaer-object shuffles the entries fed to the model
only_wt = False
db_path = '/home/lucas/esmmsa/data/stability/Meltome2.csv' #path of the database with the thermodynamic information
label_name='Tm' #what property the model tries to predict

ID_list = cluster_random(k = folds, db_path=db_path, label_name=label_name)

#%% THE MAIN LOOP
time_begin = datetime.now()
tracker = {'distr':{},
           'PCC':{},
           'MSE':{},
           'RMSE':{},
           'MAE':{},
           'R2':{}
        }

metrics = ['PCC', 'RMSE', 'MSE', 'MAE', 'R2']
for i, (train_ID, test_ID) in enumerate(ID_list):
    print("=========================")
    print(f"Fold {i}:")


    # Re-initialize model
    model = Transformer().to(device)    #<------- Change here to change model, loss-function or optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training set for fold
    train_dataset = Transformer_Dataset_precondensed(database_file=db_path,
                                    device = device,
                                    only_wt = only_wt,
                                    target_transform= Lambda (lambda y: normalize(y, typ=label_name)),
                                    selection= train_ID,
                                    label_name=label_name,
                                    prefered_sample = 'lysate')
    train_dataloader = DataLoader(dataset=train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle)

    # Validation set for fold
    val_dataset = Transformer_Dataset_precondensed(database_file=db_path,
                                    device = device,
                                    target_transform= Lambda (lambda y: normalize(y, typ=label_name)),
                                    only_wt = only_wt,
                                    selection= test_ID,
                                    label_name=label_name,
                                    prefered_sample = 'lysate')
    val_dataloader = DataLoader(dataset=val_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle)
    
    # Train and evaluate the fold
    train_loss = []
    fold_loss = []
        
    for m in metrics:
        tracker[m][i] = {'train':[], 'val':[]}
    for t in range(epochs):
        print(f"\tEpoch {t}")
        train = train_loop(train_dataloader, model, loss_fn, optimizer, device, show=True)
        val = test_loop(val_dataloader, model, loss_fn, device, show=True)
             
        for m in metrics:
            tracker[m][i]['train'].append(train[m])
            tracker[m][i]['val'].append(val[m])

    print()
    #Save info from fold
    tracker['distr'][i] = test_ID
    now = datetime.now()
    time = {'began': time_begin.strftime('%d%m%y %H%M'),
            'finished': now.strftime('%d%m%y %H%M'),
            'duration': str(now - time_begin)}
    tracker['info'] = {'model': str(model),
                       'only_wt': only_wt,
                       'learning_rate': learning_rate,
                       'epochs': epochs,
                       'batch_size': batch_size,
                       'loss_fn' : str(loss_fn),
                       'optimizer': str(optimizer),
                       'shuffle': str(shuffle),
                       'time': time,
                       'folds': i,
                       'label_name':label_name,
                       'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
                        }
    
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    with open(save_directory + "latest_run_info.json", "w") as file:
        json.dump(tracker, file)
    torch.save(model, save_directory + 'latest_model.pth')
# %%
