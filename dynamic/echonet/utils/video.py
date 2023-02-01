#!/usr/bin/env python3

"""Functions for training and running EF prediction."""
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

#import echonet

# Steven Ufkes: Added these modules
import sys
from collections import OrderedDict
import json
import argparse

import pandas as pd
import statsmodels.api as sm # only used for generateMetrics()
import random # Only added so that I can manually set the seed, in the off chance that some module uses it. Could maybe remove.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Steven Ufkes: add this line so that video.py can be called directly and import echonet module.
import echonet

def writePredictions(output, split, ds, yhat, tasks, file_list_path, subject_name_col, pixel_scale_factor_column, multiply_pixel_scale_factor_by, target_in_physical_units_column):
    """"Create a dataframe storing the subject names, predicted values averaged across subclip predictions, and predictions converted to physical units if applicable (e.g. predictions converted from pixels to cm when predicting LVOT diameter)."""    

    # Create dataframe to store actual and predicted values.
    data_df = pd.read_csv(file_list_path, index_col=subject_name_col)
    
    pred_df = pd.DataFrame(index=ds.fnames)
    pred_df.index.name='Subject'
    actual_col_name = tasks + '_actual'
    predicted_col_name = tasks + '_prediction'
    
    # Fill in the ground truth values.
    pred_df.loc[:, actual_col_name] = data_df.loc[data_df.index.isin(ds.fnames), tasks]
    pred_df.loc[:, predicted_col_name] = yhat
    
    # Save predictions to CSV.
    out_name = f'{split}-{tasks}-pred_and_actual.csv'
    out_path = os.path.join(output, out_name)
    pred_df.to_csv(out_path)
    
    # Create dataframe to store actual and predicted values in physical units, if appropriate.
    if (not pixel_scale_factor_column is None):
        # Get the the multiplicative factors used to convert raw predictions to physical units.
        if (not pixel_scale_factor_column is None):
            pred_df.loc[:, 'pixel_scale_factor'] = data_df.loc[data_df.index.isin(ds.fnames), pixel_scale_factor_column]
        if (not multiply_pixel_scale_factor_by is None):
            pred_df.loc[:, 'pixel_scale_factor'] = pred_df.loc[:, 'pixel_scale_factor'] * multiply_pixel_scale_factor_by
            
        # Convert raw predictions to phsyical units.
        actual_col_name_phys = target_in_physical_units_column + '_actual'
        predicted_col_name_phys = target_in_physical_units_column + '_prediction'            
        
        pred_df.loc[:, actual_col_name_phys] = data_df.loc[data_df.index.isin(ds.fnames), target_in_physical_units_column]
        pred_df.loc[:, predicted_col_name_phys] = pred_df.loc[:, 'pixel_scale_factor'] * pred_df.loc[:, predicted_col_name]
        
        # Remove the extra columns
        pred_df_phys = pred_df[[actual_col_name_phys, predicted_col_name_phys]]
        
        out_name = f'{split}-{target_in_physical_units_column}-pred_and_actual.csv'
        out_path = os.path.join(output, out_name)
        pred_df_phys.to_csv(out_path)
    else:
        pred_df_phys = None
    return pred_df, pred_df_phys

def calculateMetrics(df, split, run_name, target, metrics_path):
    """df is a dataframe with columns for actual and predicted values.

Should be organized better (e.g. combined with the script for LVOT diameter and EF). Should be in a separate file etc.
"""
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
    else:
        metrics_df = pd.DataFrame(columns=['run_name', 'target'], dtype=float)

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

    # Add metrics to new row
    try:
        index = metrics_df.loc[(metrics_df['run_name']==run_name) & (metrics_df['target']==target)].index[0]
    except IndexError: # if no matching row
        metrics_df = metrics_df.append(pd.Series(dtype='object'), ignore_index=True) # add row
        index = metrics_df.index[-1]
    metrics_df.loc[index, 'run_name'] = run_name
    metrics_df.loc[index, 'target'] = target
    metrics_df.loc[index, split+'-MSE'] = mse
    metrics_df.loc[index, split+'-RMSE'] = rmse
    metrics_df.loc[index, split+'-MAE'] = mae
    metrics_df.loc[index, split+'-MAPE'] = mape
    metrics_df.loc[index, split+'-R'] = r
    metrics_df.loc[index, split+'-R2'] = r2
    metrics_df.loc[index, split+'-p'] = p

    metrics_df.to_csv(metrics_path, index=False)
    return

def run(num_epochs=45,
        modelname="r2plus1d_18",
        tasks="EF",
        frames=32,
        period=2,
        pretrained=True,
        output=None,
        device=None,
        n_train_patients=None,
        num_workers=5,
        batch_size=20,
        seed=0,
        lr_step_period=15,
        run_test=False,

        #### Added by Steven Ufkes:
        # Needed within video.py
        optimizer_name="SGD",
        lr=1e-4,
        momentum=0.9,
        weight_decay=1e-4,
        return_value=None, # value returned from this function. If 'trained_model_val_loss', return the loss of the trained model in the validation set without test-time augmentation. If 'trained_model_val_loss_test_time_augmented', return the loss of the trained model in the validation set with test-time augmentation
        note='',
        run_name='',
        metrics_path = None, # path to which to save test-time performance metrics.
        start_checkpoint_path=None, # Path to checkpoint file to resume training from. Will not be overwritten unless it is in the default save location for the run.
        load_model_weights_only=False, # Resume from checkpoint, but only take the model weights etc. store in checkpoint['state_dict']. Do not get the epoch number etc. Use if you want to retrain starting from the weights trained on the EchoNet dataset.
        evaluate_parameters_path = None, # Set to None if you want to evaluate the model using the best parameters found during training. Set to another value if you want to evaluate using a specific set of model weights.
        run_train=True, # Whether or not to train the model.
        # Need to pass into echo.py
        file_list_path=None, # path to FileList.csv
        load_tracings=False, # whether to load VolumeTracings.csv
        volume_tracings_path=None, # path to VolumeTracings.csv
        file_path_col='FilePath', # Column in FileList.csv to read AVI file paths from.
        subject_name_col='Subject', # Column in FileList.csv to read subject IDs from.
        split_col='Split', # Column in FileList.csv to assign splits from.
        freeze_n_conv_layers=None, # int, how many of the (2+1D) conv layers to freeze, starting from the first layer and up to five (there are 5 (2+1D)conv layers
        set_fc_bias=True, # Original script does `model.fc.bias.data[0] = 55.6`; Let's try without doing that.
        clips=1, # Original script always uses clips=1 for training; not sure if the training loop will work with clips>1, but let us try.
        training_block_size=None, # Added this option for training when clips>1; Maximum number of augmentations to run on at the same time. Use to limit the amount of memory used. If None, always run on all augmentations simultaneously.
        #test_time_augmentation=True # When using trained model for inference, whether to perform augmentation by using every possible subclip in the input video. This may cause memory errors when input videos have many frames. NOT IMPLEMENTED
        make_plots=False, # If true, generate scatter plots. Will likely cause crashes complaining "Could not connect to any X display."
        pixel_scale_factor_column=None, # If model predictions need to be converted from pixels to physical units, perform the conversion my multiplying raw predictions by the values stored in this column of the file list. The multiplicative factors may differ between subjects.
        multiply_pixel_scale_factor_by=None, # If using a previously trained model where the pixel scale factor was incorrect by some multiplicative factor, use this value to perform the correction.
        target_in_physical_units_column=None, # If predicting a value that needs to be converted from pixels to physical units, specify the name of the column storing the target values in physical units, which will be used in performance metrics.
        ):
    """Trains/tests EF prediction model.

    Args:
        num_epochs (int, optional): Number of epochs during training
            Defaults to 45.
        modelname (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<modelname>)
            Defaults to ``r2plus1d_18''.
        tasks (str, optional): Name of task to predict. Options are the headers
            of FileList.csv.
            Defaults to ``EF''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        output (str or None, optional): Name of directory to place outputs
            Defaults to None (replaced by output/video/<modelname>_<pretrained/random>/).
        device (str or None, optional): Name of device to run on. See
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            for options. If ``None'', defaults to ``cuda'' if available, and ``cpu'' otherwise.
            Defaults to ``None''.
        n_train_patients (str or None, optional): Number of training patients. Used to ablations
            on number of training patients. If ``None'', all patients used.
            Defaults to ``None''.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 5.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 20.
        seed (int, optional): Seed for random number generator.
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            If ``None'', learning rate is not decayed.
            Defaults to 15.
        run_test (bool, optional): Whether or not to evaluate model.
            Defaults to False.
    """

    # Seed RNGs
    ## Old random number seeding, code was non-deterministic:
    #np.random.seed(seed)
    #torch.manual_seed(seed)

    ## New random number seeding, code appears deterministic when run on GPU with num_workers=0.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set default output directory
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(modelname, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    #### Steven Ufkes: Set path to checkpoint to resume from.
    if start_checkpoint_path is None:
        start_checkpoint_path = os.path.join(output, "checkpoint.pt")
    else:
        print('Warning: Resuming partially complete trains will not work if start_checkpoint_path is specified. Model will simply start training from the weights specified in start_checkpoint_path.')
    ####
    
    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.video.__dict__[modelname](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    # Manually set the bias of the fully connected layer. This was done in the original EchoNet-Dynamic code.
    if set_fc_bias:
        model.fc.bias.data[0] = 55.6

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)

    ######### Steven Ufkes: Freeze layers #########
    ### r2plus1d_18 has one child: VideoResNet
    ### VideoResNet has 7 children: 5 (2+1D) Conv layers (each possibly with slight differences), avpool, fc.
    #len(parameters) in subchild: 6
    #len(parameters) in subchild: 24
    #len(parameters) in subchild: 27
    #len(parameters) in subchild: 27
    #len(parameters) in subchild: 27
    #len(parameters) in subchild: 0
    #len(parameters) in subchild: 2

    ## Take a look at the layers.
    #print(model)
    #print('len(parameters):', len(list(model.parameters())))
    #print('len(children) in model:', len(list(model.children())))
    #for child in model.children():
        #print('len(children) in model child:', len(list(child.children())))
        #for subchild in child.children(): # 7 layers are "grandchildren
            #print(subchild)
            #print('len(parameters) in subchild:', len(list(subchild.parameters())))
            #print('len(children) in model subchild:', len(list(subchild.children())))

    # Freeze the conv layers:
    if freeze_n_conv_layers:
        if (not modelname == 'r2plus1d_18'):
            raise Exception('Freezing layers implemented only for model "r2plus1d".')
        if not (freeze_n_conv_layers in range(0,6)):
            raise Exception('freeze_n_conv_layers must be an integer in [0,1,2,3,4,5] or None')

        for child in model.children(): # 1 child
            for conv_layer in list(child.children())[:freeze_n_conv_layers]:
                for param in conv_layer.parameters():
                    param.requires_grad = False # Freeze the layer.

    ########################################################################

    model.to(device)

    # Set up optimizer
    optimizers = {
        'SGD':torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'Adam':torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False),
        'AdamW':torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    }
    optim = optimizers[optimizer_name]
    
    if lr_step_period is None:
        lr_step_period = math.inf # Steven Ufkes: I assume infinite period prevents StepLR() from decaying the learning rate.
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Steven Ufkes: Need to pass new kwargs into mean, std calculation.
    run_config_kwargs = {'file_list_path':file_list_path,
                         'load_tracings':load_tracings,
                         'volume_tracings_path':volume_tracings_path,
                         'file_path_col':file_path_col,
                         'subject_name_col':subject_name_col,
                         'split_col':split_col
                         } # Steve: I think clips should always be 1 in the mean & std calculation, so add the clips argument manually.

    dataloaders = {}

    ## Steve: If running test only, I still need the mean and standard deviation of the training data set for normalization of the test data set.
    # This should be reworked such that the mean and std are saved as model parameters. Otherwise tests require the training data.
    mean_train, std_train = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train", target_type=tasks, **run_config_kwargs), samples=None) # mean calculated on subset unless samples=None

    if run_train:
        print('Attempting to resume training from checkpoint at path: '+str(start_checkpoint_path))
        
        # Steve: They compute the mean and standard deviation of the training data (image intensity?), and use this to normalize the training data, and the test and validation data. For a test on external data, I need to normalize based on some mean and standard deviation. It seems that I should use the training mean and standard devation to normalize data acquired under the same conditions (e.g. same scanner etc.). Here, we are using a different scanner, and a scanning a pediatric sample rather than adults, so it might be sensible to normalize using the mean and standard devation of the test data.
        # Compute mean and std
        #mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))

        kwargs = {"target_type": tasks,
                  "mean": mean_train,
                  "std": std_train,
                  "length": frames,
                  "period": period,
                  }

        # Set up datasets and dataloaders
        train_dataset = echonet.datasets.Echo(split="train", **kwargs, **run_config_kwargs, pad=12, clips=clips)
        if n_train_patients is not None and len(train_dataset) > n_train_patients:
            # Subsample patients (used for ablation experiment) # Steven Ufkes: This is something from original EchoNet that we never used.
            indices = np.random.choice(len(train_dataset), n_train_patients, replace=False)
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
        dataloaders['train'] = train_dataloader

        val_dataloader = torch.utils.data.DataLoader(
            echonet.datasets.Echo(split="val", **kwargs, **run_config_kwargs, clips=clips), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
        dataloaders['val'] = val_dataloader

        test_dataloader = torch.utils.data.DataLoader(
            echonet.datasets.Echo(split="test", **kwargs, **run_config_kwargs, clips=clips), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
        dataloaders['test'] = test_dataloader

        # Run training and testing loops
        with open(os.path.join(output, "log.csv"), "a") as f:
            epoch_resume = 0
            bestLoss = float("inf")
            current_epoch_is_best = False
            try:
                # Attempt to load checkpoint

                # Steven Ufkes: Next line gives the following error if run without GPU.
                # RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.

                #checkpoint = torch.load(os.path.join(output, "checkpoint.pt")) # original line.
                checkpoint = torch.load(start_checkpoint_path)

                if device == 'cpu': # Steve: This workaround is not tested or justified; except by a suggestion online.
                    print('Steve: Loading checkpoint:', checkpoint.keys())
                    checkpoint = torch.load(start_checkpoint_path, map_location=torch.device('cpu')) # modified line.
                    print('Steve: Creating new state_dict with "module." prefixes removed. Looks like these prefixes might be an inocuous artifact of the way the checkpoint was saved. Seems like the "module." prefix is expected when a GPU is available, otherwise not. Not sure why or how it might matter.')

                    spoof_state_dict = OrderedDict()
                    for key, value in checkpoint['state_dict'].items():
                        if key.startswith('module.'):
                            new_key = key[7:]
                        spoof_state_dict[new_key] = value
                    checkpoint['state_dict'] = spoof_state_dict
                # ---- END OF WORKAROUND ----

                ## Steve: Modified the following section to only load the model weights (not the optimizer or scheduler state dict, nor the epoch number or best loss).
                model.load_state_dict(checkpoint['state_dict'])
                if not load_model_weights_only:
                    print('Steve: Loading state_dict for model, optim, and scheduler; loading epoch number and bestLoss.')
                    optim.load_state_dict(checkpoint['opt_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_dict'])
                    epoch_resume = checkpoint["epoch"] + 1
                    bestLoss = checkpoint["best_loss"]
                else:
                    print('Steve: Loaded model.state_dict() only. Epoch, optimizer, scheduler, and bestLoss of checkpoint ignored.')
                # ---- END OF MODIFICATION ----

                f.write("Resuming from epoch {}\n".format(epoch_resume))
            except FileNotFoundError:
                f.write("Starting training from scratch\n")

            for epoch in range(epoch_resume, num_epochs):
                print("Epoch #{}".format(epoch), flush=True)
                for phase in ['train', 'val', 'test']:
                    start_time = time.time()
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_max_memory_allocated(i)
                        torch.cuda.reset_max_memory_cached(i)
                    ## NB: In the next line, run_epoch() will update the model only if phase=='train'.
                    loss, yhat, y = echonet.utils.video.run_epoch(model, dataloaders[phase], phase=="train", optim, device, block_size=training_block_size)
                    f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                  phase,
                                                                  loss,
                                                                  sklearn.metrics.r2_score(yhat, y),
                                                                  time.time() - start_time,
                                                                  y.size,
                                                                  sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                  sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                                  batch_size))
                    if phase == 'val':
                        val_loss = loss
                        if val_loss < bestLoss:
                            bestLoss = val_loss
                            current_epoch_is_best = True
                    f.flush()

                scheduler.step()

                # Save checkpoint
                save = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'period': period,
                    'frames': frames,
                    'best_loss': bestLoss,
                    'loss': val_loss,
                    'r2': sklearn.metrics.r2_score(yhat, y),
                    'opt_dict': optim.state_dict(),
                    'scheduler_dict': scheduler.state_dict(),
                }
                torch.save(save, os.path.join(output, "checkpoint.pt"))
                if current_epoch_is_best:
                    torch.save(save, os.path.join(output, "best.pt"))
                    current_epoch_is_best = False

            # Record information about best epoch to log.
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"])) # Commented out because log file is no longer open.
            f.flush()

    if run_test:
        ## Load trained model weights.
        if evaluate_parameters_path is None:
            evaluate_parameters_path = os.path.join(output, 'best.pt')
        checkpoint = torch.load(evaluate_parameters_path)
        model.load_state_dict(checkpoint['state_dict'])

        with open(os.path.join(output, 'log.csv'), 'a') as f:
            for split in ['train', 'val', 'test']:
                # Steve: They compute the mean and standard devation of the training data (image intensity?), and use this to normalize the training data, and the test and validation data. For a test on external data, I need to normalize based on some mean and standard deviation. It seems that I should use the training mean and standard devation to normalize data acquired under the same conditions (e.g. same scanner etc.). Here, we are using a different scanner, and a scanning a pediatric sample rather than adults, so it might be sensible to normalize using the mean and standard devation of the test data.

                # Compute mean and std
                kwargs = {"target_type": tasks,
                          "mean": mean_train,
                          "std": std_train,
                          "length": frames,
                          "period": period,
                          }

                # Performance without test-time augmentation
                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(split=split, **kwargs, **run_config_kwargs), # Steve: clips=1 default will be used here.
                    batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device)
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Performance with test-time augmentation
                ds = echonet.datasets.Echo(split=split, **kwargs, **run_config_kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=50)
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()

                if split=='val':
                    trained_model_val_loss_test_time_augmented = tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))[0]

                # Write performance across all subclips to file.
                with open(os.path.join(output, "{}_predictions-all_subclips.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                echonet.utils.latexify()


                #### Steven Ufkes update 2023-01-31:
                # Make dataframe storing subject names and predictions averaged across subclips. If predicting LVOT diameter, store predictions in units of pixels and cm.
                yhat = np.array(list(map(lambda x: x.mean(), yhat))) # predictions averaged across subclips.
                pred_df, pred_df_phys = writePredictions(output, split, ds, yhat, tasks, file_list_path, subject_name_col, pixel_scale_factor_column, multiply_pixel_scale_factor_by, target_in_physical_units_column)

                # Record performance metrics for trained model.
                calculateMetrics(pred_df, split, run_name, tasks, metrics_path)
                if (not pred_df_phys is None):
                    calculateMetrics(pred_df_phys, split, run_name, target_in_physical_units_column, metrics_path)
                #### END OF UPDATE 2023-01-31 ####
                
                # Plot actual and predicted EF
                if make_plots:
                    fig = plt.figure(figsize=(3, 3))
                    lower = min(y.min(), yhat.min())
                    upper = max(y.max(), yhat.max())
                    plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                    plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                    plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                    plt.gca().set_aspect("equal", "box")
                    plt.xlabel("Actual EF (%)")
                    plt.ylabel("Predicted EF (%)")
                    plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                    plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                    plt.close(fig)

    # Return a performance metric value for hyperparameter optimization.
    if return_value == 'trained_model_val_loss':
        return bestLoss
    elif return_value == 'trained_model_val_loss_test_time_augmented':
        return trained_model_val_loss_test_time_augmented
    elif return_value is None:
        return

def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """
    model.train(train)

    total = 0  # total training loss
    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader), disable=True) as pbar:
            for (X, outcome) in dataloader:

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape # Steve: Not sure what batch is.
                    X = X.view(-1, c, f, h, w) # Steve: has dimensions (?, color, frames, height, width)

                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j+block_size), ...]) for j in range(0, X.shape[0], block_size)])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y

## Steven Ufkes: Add command line wrapper allowing the passing of a configuration file instead of arguments.
def run_from_config(arg_dict=None):
    # Usually a configuration JSON file is passed as an argument to run.py from the command line. Alternatively, call this function directly passing in a Python dictionary containing the run configuration (e.g. for automatic hyperparameter optimization).
    if arg_dict is None: # If running from command line, argdict will be None.
        # Create argument parser.
        description = """Execute video.run with arguments specified in a JSON file."""
        parser = argparse.ArgumentParser(description=description)

        # Define positional arguments.
        parser.add_argument("config_path", help="JSON configuration file containing arguments accepted by video.run().", type=str)
        
        # Parse arguments.
        config_args = parser.parse_args()

        # Read arguments to dict.
        with open(config_args.config_path, 'r') as f:
            arg_dict = json.load(f)

    # Copy config to output directory if specified.
    if 'output' in arg_dict:
        os.makedirs(arg_dict['output'], exist_ok=True)
        with open(os.path.join(arg_dict['output'], 'config.json'), 'w') as f:
            json.dump(arg_dict, f, indent=4)
            print("Starting run with configuration:")
            print(json.dumps(arg_dict,  indent=4))

    # Run main function.
    return run(**arg_dict)

if __name__ == '__main__':
    run_from_config()
