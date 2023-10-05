#! /home/lucas/miniconda3/envs fresh
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import json
from helper_functions import *


# %% Parameters and stuff to change
device = torch.device('cuda:0')
learning_rate = 0.000105
batch_size = 64
weight_decay= 0.0003
shuffle = True  #to more properly make training reproducible should be false.
only_wt = False
directory = '/home/lucas/esmmsa/data/stability/Meltome2.csv'
label_name = 'Tm' #change to dG if you want to train on difference of Gibbs free energy

def train_cifar(config, train_ID, val_ID):
    model = RNN(rnn_hidden=config['rnn_hidden'],
                 mlp2_hidden=config['mlp2_hidden'],
                 dropout_in=0.4, 
                 dropout_lstm=0.3, 
                 dropout_out=0.2)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate, weight_decay = weight_decay)
    
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # Training set for fold
    train_dataset = RNN_Dataset(database_file=directory,
                                    device = device,
                                    only_wt = only_wt,
                                    target_transform= Lambda (lambda y: normalize(y, typ=label_name)),
                                    selection= train_ID,
                                    label_name = label_name)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle,
                                    collate_fn=collate_batch)

    # Testing set for fold
    val_dataset = RNN_Dataset(database_file=directory,
                                    device = device,
                                    target_transform= Lambda (lambda y: normalize(y, typ=label_name)),
                                    only_wt = only_wt,
                                    selection= val_ID,
                                    label_name = label_name,
                                    mode='val')
    val_dataloader = DataLoader(dataset=val_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle,
                                    collate_fn=collate_batch)



    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, (emb, attent, y, _) in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            emb, attent, y = emb.to(device), attent.to(device), y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(emb, attent)

            y = y.float().to(device)
            y = torch.reshape(y, (y.shape[0], 1))

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 100 == 99:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, (emb, attent, y, _) in enumerate(val_dataloader, 0):
            with torch.no_grad():
                emb, attent, y = emb.to(device), attent.to(device), y.to(device)
                y = y.float()
                y = torch.reshape(y, (y.shape[0], 1))

                outputs = model(emb, attent)
                total += y.size(0)

                loss = criterion(outputs, y)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps},
            checkpoint=checkpoint,
        )
    print("Finished Training")    


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        'rnn_hidden': tune.choice([128, 256, 512]),
        'mlp2_hidden':tune.choice([]),
        'dropout_out':tune.choice([0.1, 0.4, 0.6]),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=3, #number of epochs before abandoning the poorly performing ones
        reduction_factor=2,
    )

    df = pd.read_csv(directory)
    train_ID = df.loc[df['set'] == 'training', 'name'].tolist()
    val_ID = df.loc[df['set'] == 'validation', 'name'].tolist()

    result = tune.run(
        partial(train_cifar, train_ID=train_ID, val_ID=val_ID),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler
        )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    best_trained_model = RNN
    device = "cuda:0"
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
        # if gpus_per_trial > 1:
        #     best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])
    os.chdir('/home/lucas/esmmsa/networks/models/')
    torch.save(best_trained_model, 'best_performing_model.pth')
    print(f"COMPLETELY FINISHED.")
    
    tracker = {'info':{}}
    tracker['info'] = {'model': str(best_trained_model),
                       'weight_decay': best_trial.config['weight_decay'],
                       'dropout_in': best_trial.config['dropout_in'],
                       'dropout_transformer': best_trial.config['dropout_transformer'],
                       'dropout_out':best_trial.config['dropout_out'],
                       'only_wt': only_wt,
                       'epochs': max_num_epochs,
                       'learning_rate': best_trial.config['learning_rate'],
                       'batch_size': batch_size,
                       'loss_fn' : 'MSELoss',
                       'optimizer': 'AdamW',
                       'shuffle': str(shuffle),
                       'num_parameters': sum(p.numel() for p in best_trained_model.parameters() if p.requires_grad)
                        }
    with open("/home/lucas/esmmsa/networks/models/best_performance_config.json", "w") as file:
        json.dump(tracker, file)

    

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=20, gpus_per_trial=2)