#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import net
from dataset import DopplerDataset

def train_loop(dataloader, model, loss_fn, optimizer, device, verbose=False, train_noise=0.0):
    size = len(dataloader.dataset)
    total_loss = 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        if train_noise > 0:
            y += np.random.normal(loc=0.0, scale=train_noise)
        y = y.to(device) # Shape = (batch_size, 1)
        
        # Compute prediction and loss
        pred = model(X)
        pred = nn.Flatten(start_dim=0)(pred)
        
        loss = loss_fn(pred, y) # MSE of batch

        #print(y, pred, loss)

        #print(X[0].shape, X[0].mean(), X[0].std())

        # Backpropagation
        optimizer.zero_grad() # Set the gradients back to zero (from the previous loop?).
        loss.backward() # Compute the gradients.
        optimizer.step() # Update the parameters.

        total_loss += loss.item() * y.shape[0] # (MSE of batch) * (size of batch) = SSE of batch
        
    mean_loss = total_loss/size
    if verbose:
        print(f"Mean loss (training)  : {mean_loss:>8f}")

    return mean_loss

def val_loop(dataloader, model, loss_fn, device, verbose=False):
    size = len(dataloader.dataset)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            pred = nn.Flatten(start_dim=0)(pred)
            
            loss = loss_fn(pred, y)
            #total_loss += loss.item()
            
            total_loss += loss.item() * y.shape[0] # (MSE of batch) * (size of batch) = SSE of batch

    mean_loss = total_loss/size
    if verbose:
        print(f"Mean loss (validation) : {mean_loss:>8f}")

    return mean_loss

## Try a custom replacement of the standard validation loop above.
def val_loop_custom(dataloader, model, loss_fn, device, verbose=False):
    size = len(dataloader.dataset)
    total_loss = 0

    dataset = dataloader.dataset
    
    model.eval()
    with torch.no_grad():
        for index, (img, y) in enumerate(dataloader): # works if batch_size=1 and shuffle=False
            subject = dataset.data_df.loc[index, 'Subject']
                
            #img, y = dataset[index]
            img = img.to(device)
            yhat = model(img)

            ## Convert values back to cm/s.
            # Prediction is for a number of pixels in the image (divided by downscale_y = 10000).
            # Each pixel represents a certain number of centimeters, different for each subject.
            scale_factor = dataset.downscale_y * dataset.data_df.loc[index, 'pixel_scale_factor']
            y = y.item() * scale_factor
            yhat = yhat.item() * scale_factor

            loss = (y - yhat)**2
            total_loss += loss
        
    mean_loss = total_loss/size
    if verbose:
        print(f"Mean loss (validation) : {mean_loss:>8f}")

    return mean_loss

def main(seed=489, out_dir='../runs/doppler'):

    ## Set seeds for determinism.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    #random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    #### Set hyperparameters
    ## Model settings:
    freeze_layers = 2 # For resnet18, let's try 0, 2, 5, 6, 7, 8 (want to freeze batch norm layers following each conv layer (I think)
    dropout_rate = 0.0

    ## Training loop settings:
    optimizer_name = 'AdamW'
    learning_rate = 3.0e-4 # 2.5e-4 paper
    weight_decay = 0.008 # 0.008 paper
    epochs = 200 # 50 paper
    train_noise = 0.0 # 0.03 paper

    ## Dataloader settings:
    file_list_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti.csv'
    batch_size = 48
    shuffle = True
    drop_last = False
    normalize_mode = 'training_set'
 
    ## Other
    note = 'redo best paper run - will likely be different result due to changes in normalization method'
    
    #### Select device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Using device:', device)
    
    #### Make datasets.
    train_dataset = DopplerDataset(split='train', file_list_path=file_list_path, normalize_mode=normalize_mode)
    val_dataset = DopplerDataset(split='val', file_list_path=file_list_path, normalize_mode=normalize_mode)
    test_dataset = DopplerDataset(split='test', file_list_path=file_list_path, normalize_mode=normalize_mode)

    ## Record the image dimensions.
    dim_x = train_dataset[0][0].shape[1]
    dim_y = train_dataset[0][0].shape[2]
    
    ## Make dataloaders.
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=drop_last)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=1, # must be set to one for custom validation loop. Maybe change this back to a separate dataloader.
                                shuffle=False, # never a point in shuffling the validation set, right?
                                drop_last=False)
#    val_dataloader = DataLoader(val_dataset,
#                                batch_size=batch_size,
#                                shuffle=shuffle,
#                                drop_last=drop_last)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=False)


    #### Set model and hyperparameters
    ## Set model.
    model = net.myResNet18(freeze_layers=freeze_layers, dropout_rate=dropout_rate)
    model.to(device)

    ## Set loss function.
    loss_fn = nn.MSELoss()

    ## Set optimizer.
    optimizers = {'SGD':torch.optim.SGD(model.parameters(), lr=learning_rate),
                  'Adam':torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False),
                  'AdamW':torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)}
    optimizer = optimizers[optimizer_name]
    

    ## Create spreadsheet to store loss vs. epoch
    loss_df = pd.DataFrame(columns=['train', 'val', 'test'])
    loss_df.index.name = 'epoch'
    
    #### Training and validation loop
    val_mean_loss_min = np.inf
    for t in range(epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        print(f"\nEpoch {t+1}")
        train_mean_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device, train_noise=train_noise)
        val_mean_loss = val_loop(val_dataloader, model, loss_fn, device)
        val_mean_loss_custom = val_loop_custom(val_dataloader, model, loss_fn, device) # ideally use this to determine best epoch
        test_mean_loss = val_loop(test_dataloader, model, loss_fn, device)

        loss_df.loc[t+1, ['train', 'val', 'test']] = [train_mean_loss, val_mean_loss, test_mean_loss]

        print(f"Training         : {train_mean_loss}\nValidation       : {val_mean_loss}\nValidation (real): {val_mean_loss_custom}")
        
        if val_mean_loss < val_mean_loss_min:
            val_mean_loss_min = val_mean_loss
            val_mean_loss_custom_at_val_min = val_mean_loss_custom
            train_mean_loss_at_val_min = train_mean_loss
            best_epoch = t+1
            print(f"* * * New best validation result * * *")
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
            
    print(f"Done training.\nBest validation loss: {val_mean_loss_min} ({val_mean_loss_custom_at_val_min})")

    ## Save loss vs. epoch
    loss_name = 'loss.csv'
    loss_path = os.path.join(out_dir, loss_name)
    loss_df.to_csv(loss_path, index=True)
    
    ## Save hyperparameters and losses to a spreadsheet.
    hyper_name = 'hyperparameters.csv'
    hyper_path = os.path.join(out_dir, hyper_name)
    if os.path.exists(hyper_path):
        hyper_df = pd.read_csv(hyper_path)
    else:
        hyper_df = pd.DataFrame()
    hyper_df = hyper_df.append(pd.Series(dtype='object'), ignore_index=True)
    hyper_df.loc[hyper_df.index[-1], 'dim_x'] = dim_x
    hyper_df.loc[hyper_df.index[-1], 'dim_y'] = dim_y
    hyper_df.loc[hyper_df.index[-1], 'note'] = note
    hyper_df.loc[hyper_df.index[-1], 'epochs'] = epochs
    hyper_df.loc[hyper_df.index[-1], 'train_noise'] = train_noise
    hyper_df.loc[hyper_df.index[-1], 'learning_rate'] = learning_rate
    hyper_df.loc[hyper_df.index[-1], 'weight_decay'] = weight_decay
    hyper_df.loc[hyper_df.index[-1], 'batch_size'] = batch_size
    hyper_df.loc[hyper_df.index[-1], 'shuffle'] = shuffle
    hyper_df.loc[hyper_df.index[-1], 'drop_last'] = drop_last
    hyper_df.loc[hyper_df.index[-1], 'normalize_mode'] = normalize_mode
    hyper_df.loc[hyper_df.index[-1], 'optimizer'] = optimizer_name
    hyper_df.loc[hyper_df.index[-1], 'freeze_layers'] = freeze_layers
    hyper_df.loc[hyper_df.index[-1], 'dropout_rate'] = dropout_rate
    hyper_df.loc[hyper_df.index[-1], 'best_epoch'] = best_epoch
    hyper_df.loc[hyper_df.index[-1], 'mse_mean_val'] = val_mean_loss_min
    hyper_df.loc[hyper_df.index[-1], 'mse_mean_val_real'] = val_mean_loss_custom_at_val_min
    hyper_df.loc[hyper_df.index[-1], 'mse_mean_train'] = train_mean_loss_at_val_min


    # Reorder columns so that losses are last
    end_cols = ['best_epoch', 'mse_mean_val', 'mse_mean_val_real', 'mse_mean_train']
    cols = [c for c in hyper_df.columns if not c in end_cols] + end_cols
    hyper_df = hyper_df[cols]
    hyper_df.to_csv(hyper_path, index=False)    
    
    ## Generate tables of actual and predicted values.
    best_state_dict = torch.load(os.path.join(out_dir, 'best.pt'))
    model.load_state_dict(best_state_dict)
    # Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results (https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).

    target_type = train_dataset.target_type # e.g. AOVTI_px

    dataset_dict = {}
    dataset_dict['train'] = train_dataset
    dataset_dict['val'] = val_dataset
    #dataset_dict['test'] = test_dataset

    model.eval()
    with torch.no_grad():
        for split, dataset in dataset_dict.items():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            df = pd.DataFrame(columns=[target_type+'_actual', target_type+'_prediction'])
            df.index.name = 'Subject'

            df_raw = pd.DataFrame(columns=[target_type+'_raw_actual', target_type+'_raw_prediction'])
            df_raw.index.name = 'Subject'
            #for index in range(len(dataset)):
            for index, (img, y) in enumerate(dataloader):
                subject = dataset.data_df.loc[index, 'Subject']
                
                #img, y = dataset[index]
                img = img.to(device)
                yhat = model(img)

                # Save the predictions in units of pixels for sanity check.
                df_raw.loc[subject, target_type+'_raw_actual'] = y.item()
                df_raw.loc[subject, target_type+'_raw_prediction'] = yhat.item()
                
                ## Convert values back to cm/s.
                # Prediction is for a number of pixels in the image (divided by downscale_y = 10000).
                # Each pixel represents a certain number of centimeters, different for each subject.
                scale_factor = dataset.downscale_y * dataset.data_df.loc[index, 'pixel_scale_factor']
                y = y.item() * scale_factor
                yhat = yhat.item() * scale_factor 
                
                df.loc[subject, target_type+'_actual'] = y.item()
                df.loc[subject, target_type+'_prediction'] = yhat.item()

            out_name = split+ '-' + target_type + '-pred_and_actual.csv'
            out_path = os.path.join(out_dir, out_name)
            df.to_csv(out_path, index=True)
            
            #out_name = split+ '-' + target_type + '-raw-pred_and_actual.csv'
            #out_path = os.path.join(out_dir, out_name)
            #df_raw.to_csv(out_path, index=True)
    

if __name__ == '__main__':
    main()
