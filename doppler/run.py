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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image

import statsmodels.api as sm # only used for generateMetrics(), which should be moved out of this file.

def train_loop(dataloader, model, loss_fn, optimizer, device, verbose=False, train_noise=0.0):
    size = len(dataloader.dataset)
    total_loss = 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        if train_noise > 0:
            y += np.random.normal(loc=0.0, scale=train_noise)
        y = y.to(device) # shape = (batch_size, out_features) if out_features > 1, else (batchsize, ) ?
        current_batch_size = y.shape[0]

        # Compute prediction and loss
        pred = model(X) # shape = (batch_size, out_features)
        

        ## Seems like flattening is only necessary when the number of output features is 1, in which case y.shape = (batch_size, ), while yhat.shape = (batch_size, 1). Do a sketchy case-dependent flatten for now. When predicting AOVTI, I had only been flatting yhat. 
        if (y.shape != pred.shape):
            y = nn.Flatten(start_dim=0)(y)
            pred = nn.Flatten(start_dim=0)(pred)

        loss = loss_fn(pred, y) # MSE of batch

        # Backpropagation
        optimizer.zero_grad() # Set the gradients back to zero (from the previous loop?).
        loss.backward() # Compute the gradients.
        optimizer.step() # Update the parameters.

        total_loss += loss.item() * current_batch_size # (MSE of batch) * (size of batch) = SSE of batch
        
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

            current_batch_size = y.shape[0]
            
            pred = model(X)

            ## Seems like flattening is only necessary when the number of output features is 1, in which case y.shape = (batch_size, ), while yhat.shape = (batch_size, 1). Do a sketchy case-dependent flatten for now.
            if (y.shape != pred.shape):
                y = nn.Flatten(start_dim=0)(y)
                pred = nn.Flatten(start_dim=0)(pred)
            
            loss = loss_fn(pred, y)
            
            total_loss += loss.item() * current_batch_size # (MSE of batch) * (size of batch) = SSE of batch

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

def calculateMetrics(df, split, run_name, metrics_out_path='../runs/doppler/metrics-doppler.csv'):
    """df is a dataframe with columns for acutal and predicted values.

Should be organized better (e.g. combined with the script for LVOT diameter and EF). Should be in a separate file etc.
"""
    if os.path.exists(metrics_out_path):
        metrics_df = pd.read_csv(metrics_out_path, index_col='run_name')
    else:
        metrics_df = pd.DataFrame(dtype=float)
        metrics_df.index.name = 'run_name'

    actual_col = [col for col in df.columns if col.endswith('actual')][0]
    prediction_col = [col for col in df.columns if col.endswith('prediction')][0] # First column ending in 'prediction'

    actual = np.array(df[actual_col])
    prediction = np.array(df[prediction_col])
    
    mean_actual = actual.mean()
    
    n = len(actual)
    assert n == len(prediction)

    ## Compute performance metrics.
    # MSE
    mse = 1/n*((actual-prediction)**2).sum()
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = 1/n*(np.abs(actual-prediction)).sum()

    # MAPE
    mape = 100/n*np.abs((actual-prediction)/actual).sum()
            
    # R
    r = np.corrcoef(actual, prediction)[1,0]

    # R squared
    r2 = r**2

    # p-value - doesn't seem to work when p < 1e-12 or so; gives crazy numbers like 1e-80
    y = df[prediction_col]
    x = df[actual_col]
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    fit = model.fit()
    p = fit.pvalues[1]

    metrics_df.loc[run_name, split+'-MSE'] = mse
    metrics_df.loc[run_name, split+'-RMSE'] = rmse
    metrics_df.loc[run_name, split+'-MAE'] = mae
    metrics_df.loc[run_name, split+'-MAPE'] = mape
    metrics_df.loc[run_name, split+'-R'] = r
    metrics_df.loc[run_name, split+'-R2'] = r2
    metrics_df.loc[run_name, split+'-p'] = p

    metrics_df.to_csv(metrics_out_path, index=True)
    return

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
    epochs = 5 # 50 paper
    train_noise = 0.0 # 0.03 paper

    ## Dataloader settings:
    file_list_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti-split_peaks-417x286-rescale-with_peak_arrays.csv'
    #file_list_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti-split_peaks-417x286-pad-with_peak_arrays.csv'
    #file_list_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti-good.csv' # doesn't have train_mean or train_std
    batch_size = 48
    shuffle = True
    drop_last = False
    #normalize_mode = 'training_set'
    normalize_mode = 'self'
    target_type = 'peak_array'
    #target_type = 'AOVTI_px'
    
    ## Other
    note = 'weights from 400-epoch peak array prediction on rescaled images; first test of peak-mean VTI predictions'
    
    train = False
    predict = True

    #### Select device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Using device:', device)
    
    #### Make datasets.
    train_dataset = DopplerDataset(split='train', target_type=target_type, file_list_path=file_list_path, normalize_mode=normalize_mode)
    val_dataset = DopplerDataset(split='val', target_type=target_type, file_list_path=file_list_path, normalize_mode=normalize_mode)
    test_dataset = DopplerDataset(split='test', target_type=target_type, file_list_path=file_list_path, normalize_mode=normalize_mode)

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

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=False)


    #### Set model and hyperparameters
    ## Set model.
    if target_type == 'peak_array':
        # If predicting the VT curve, need to determine size of model output. Do by checking size of first peak array.
        out_features = len(train_dataset[0][1])
    else:
        out_features = 1
    model = net.myResNet18(freeze_layers=freeze_layers, dropout_rate=dropout_rate, out_features=out_features)
    model.to(device)


    if train:
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
            if target_type == 'AOVTI_px':
                val_mean_loss_custom = val_loop_custom(val_dataloader, model, loss_fn, device)
            test_mean_loss = val_loop(test_dataloader, model, loss_fn, device)

            loss_df.loc[t+1, ['train', 'val', 'test']] = [train_mean_loss, val_mean_loss, test_mean_loss]

            if target_type == 'AOVTI_px':
                print(f"Training         : {train_mean_loss}\nValidation       : {val_mean_loss}\nValidation (real): {val_mean_loss_custom}")
            else:
                print(f"Training         : {train_mean_loss}\nValidation       : {val_mean_loss}")
            
            if val_mean_loss < val_mean_loss_min:
                val_mean_loss_min = val_mean_loss
                if target_type == 'AOVTI_px':
                    val_mean_loss_custom_at_val_min = val_mean_loss_custom
                train_mean_loss_at_val_min = train_mean_loss
                best_epoch = t+1
                print(f"* * * New best validation result * * *")
                torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))

        if target_type == 'AOVTI_px':
            print(f"Done training.\nBest validation loss: {val_mean_loss_min} ({val_mean_loss_custom_at_val_min})")
        else:
            print(f"Done training.\nBest validation loss: {val_mean_loss_min}")

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
        hyper_df.loc[hyper_df.index[-1], 'target_type'] = target_type
        hyper_df.loc[hyper_df.index[-1], 'normalize_mode'] = normalize_mode
        hyper_df.loc[hyper_df.index[-1], 'optimizer'] = optimizer_name
        hyper_df.loc[hyper_df.index[-1], 'freeze_layers'] = freeze_layers
        hyper_df.loc[hyper_df.index[-1], 'dropout_rate'] = dropout_rate
        hyper_df.loc[hyper_df.index[-1], 'best_epoch'] = best_epoch
        hyper_df.loc[hyper_df.index[-1], 'mse_mean_val'] = val_mean_loss_min
        if target_type == 'AOVTI_px':
            hyper_df.loc[hyper_df.index[-1], 'mse_mean_val_real'] = val_mean_loss_custom_at_val_min
        hyper_df.loc[hyper_df.index[-1], 'mse_mean_train'] = train_mean_loss_at_val_min


        # Reorder columns so that losses are last
        end_cols = ['best_epoch', 'mse_mean_val', 'mse_mean_val_real', 'mse_mean_train']
        cols = [c for c in hyper_df.columns if not c in end_cols] + end_cols
        hyper_df = hyper_df[cols]
        hyper_df.to_csv(hyper_path, index=False)    

    if predict:
        ## Generate tables of actual and predicted values.
        best_state_dict = torch.load(os.path.join(out_dir, 'best.pt'))
        model.load_state_dict(best_state_dict)
        # Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results (https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).

        dataset_dict = {}
        dataset_dict['train'] = train_dataset
        dataset_dict['val'] = val_dataset
        #dataset_dict['test'] = test_dataset

        model.eval()
        with torch.no_grad():
            for split, dataset in dataset_dict.items():
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

                df = pd.DataFrame(columns=[target_type+'_actual', target_type+'_prediction'], dtype=float)
                df.index.name = 'Subject'
                
                if target_type == 'peak_array':
                    ## Make directory to store predicted peak curves.
                    plot_out_dir = os.path.join(out_dir, 'peak_curves')
                    os.makedirs(plot_out_dir, exist_ok=True)

                for index, (img, y) in enumerate(dataloader):
                    subject = dataset.data_df.loc[index, 'Subject']
                
                    img = img.to(device)
                    yhat = model(img)

                    # Load actual and predicted values. This will be a single number if target_type == 'AOVTI', and a vector if target_type == 'peak_array'
                    y = y[0].cpu().numpy()
                    yhat = yhat[0].cpu().numpy()

                    if target_type == 'peak_array':
                        ## Generate Doppler image with peak curves overlaid.
                        original_image_path = dataset.data_df.loc[index, 'FilePath']
                        
                        original_image = Image.open(original_image_path)
                        pixel_data = np.array(original_image)
                        if len(pixel_data.shape) == 2: # if in greyscale, convert to RGB.
                            pixel_data = np.stack([pixel_data]*3, axis=2) # shape [H, W, C=3]
                        for x, (y_val, yhat_val) in enumerate(zip(y.astype(int), yhat.astype(int))):
                            pixel_data[yhat_val:min(pixel_data.shape[0], yhat_val+2), x, 0] = 255 # mark the predicted peak in red
                            pixel_data[y_val:min(pixel_data.shape[0], y_val+2), x, 1] = 255 # mark the ground truth peak in green
                        image_with_peaks = Image.fromarray(pixel_data, mode='RGB')
                        image_out_dir = os.path.join(plot_out_dir, split)
                        os.makedirs(image_out_dir, exist_ok=True)
                        out_path = os.path.join(image_out_dir, subject+'.png')
                        image_with_peaks.save(out_path)

                        ## Store the VTI prediction for the curve; will later be combined with predictions for other peaks to compute subject VTI.
                        subject_base = dataset.data_df.loc[index, 'Subject_base']
                        df.loc[subject, 'Subject_base'] = subject_base # for peak images, subject will have a name like VTI123_5, and subject_base will be the orignal subject name, e.g. VTI123

                        vti = y.sum()
                        vti_hat = yhat.sum()

                        ## Convert from units of pixels back to cm.
                        # Prediction is for a number of pixels in the image. 
                        # Each pixel represents a certain number of centimeters, different for each subject.
                        scale_factor = dataset.data_df.loc[index, 'pixel_scale_factor'] # do not use downscale_y for the target_type == 'peak_array'
                        vti_cm = vti * scale_factor
                        vti_hat_cm = vti_hat * scale_factor
                        
                        df.loc[subject, target_type+'_actual'] = vti_cm
                        df.loc[subject, target_type+'_prediction'] = vti_hat_cm

                    else:
                        ## Convert from units of pixels back to cm.
                        # Prediction is for a number of pixels in the image (divided by downscale_y = 10000).
                        # Each pixel represents a certain number of centimeters, different for each subject.
                        scale_factor = dataset.downscale_y * dataset.data_df.loc[index, 'pixel_scale_factor']
                        #y_cm = y.item() * scale_factor # before adding cpu().numpy()
                        #yhat_cm = yhat.item() * scale_factor # before adding cpu().numpy()
                        y_cm = y * scale_factor # actual VTI in units of cm
                        yhat_cm = yhat[0] * scale_factor # predicted VTI in units of cm

                        # Save the predictions in units of pixels for sanity check.
                        df.loc[subject, target_type+'_actual'] = y_cm # before adding cpu().numpy()
                        df.loc[subject, target_type+'_prediction'] = yhat_cm # before adding cpu().numpy()
                        
                        #df_raw.loc[subject, target_type+'_raw_actual'] = y.item()
                        #df_raw.loc[subject, target_type+'_raw_prediction'] = yhat.item()

                if target_type == 'peak_array':
                    # Compute the mean VTI across all peaks for the subject.
                    groupby_subject_base = df.groupby('Subject_base')
                    mean_df = groupby_subject_base.mean()
                    std_df = groupby_subject_base.std()
                    median_df = groupby_subject_base.median()
                    
                    out_name = split + '-' + target_type + '-pred_and_actual.csv'
                    out_path = os.path.join(out_dir, out_name)
                    df.to_csv(out_path, index=True)

                    ## Sketchy metrics calculation. Should be reorganized and probably combined wiit metric calculation for EF and LVOT diameter.
                    ## Combine predictions across peaks in different ways, calculating performance metrics for each method separately.
                    # mean
                    aggregation_type = 'mean'
                    run_name = f'target_type: {target_type}; aggregation_type: {aggregation_type} - {note}'
                    calculateMetrics(mean_df, split, run_name=run_name)
                    predictions_out_name = f'{split}-{target_type}-{aggregation_type}-pred_and_actual.csv'
                    predictions_out_path = os.path.join(out_dir, predictions_out_name)
                    mean_df.to_csv(predictions_out_path, index=True)

                    # median
                    aggregation_type = 'median'
                    run_name = f'target_type: {target_type}; aggregation_type: {aggregation_type} - {note}'
                    calculateMetrics(median_df, split, run_name=run_name)
                    predictions_out_name = f'{split}-{target_type}-{aggregation_type}-pred_and_actual.csv'
                    predictions_out_path = os.path.join(out_dir, predictions_out_name)
                    median_df.to_csv(predictions_out_path, index=True)
                    
                    
                else:
                    out_name = split + '-' + target_type + '-pred_and_actual.csv'
                    out_path = os.path.join(out_dir, out_name)
                    df.to_csv(out_path, index=True)

                    #out_name = split+ '-' + target_type + '-raw-pred_and_actual.csv'
                    #out_path = os.path.join(out_dir, out_name)
                    #df_raw.to_csv(out_path, index=True)

                    # Calculate performance metrics.
                    run_name = f'target_type: {target_type} - {note}'
                    calculateMetrics(df, split, run_name=run_name)

if __name__ == '__main__':
    main()
