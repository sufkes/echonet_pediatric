#!/usr/bin/env python

import os
import sys

import hyperopt

from run import run_from_config

## Define the objective function.
def objective(hyperparameters):
    print(hyperparameters)
    # train_and_evaluate() expects a bunch of keyword arguments. Replace those which I want to optimize with the values in params. Set others to fixed values.
    args = {
        'run_name':f'hyperopt-loss_val_mse-lr_{hyperparameters["learning_rate"]:.5f}-wd_{hyperparameters["weight_decay"]:.5f}-tn_{hyperparameters["train_noise"]:.1f}-batch_size_{hyperparameters["batch_size"]}-force_positive_velocity_{hyperparameters["force_positive_velocity"]}',
        'out_dir':None, # if None, automatically set to <time stamp>-<run_name>
        'note':'',

        'hyper_path':'/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/doppler/hyperparameters.csv', # path to which to save hyperparameters and training loss values.
        'metrics_path':'/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/doppler/metrics-vti_camerons_annotations.csv', # path to which to save test-time performance metrics.
         
        ## Model settings:
        'freeze_layers':2, # For resnet18, let's try 0, 2, 5, 6, 7, 8 (want to freeze batch norm layers following each conv layer (I think)
        'dropout_rate':0.0,
        'initial_parameters_path':'/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/doppler/peak_curves-steves_annotations/peak_curves-rescale/best.pt', # Set to None if starting from scratch.
        'seed':489,

        ## Training loop settings:
        'optimizer_name':'AdamW',
        'learning_rate':hyperparameters['learning_rate'],
        'weight_decay':hyperparameters['weight_decay'],
        'epochs':80,
        'train_noise':hyperparameters['train_noise'],

        ## Dataloader settings:
        'file_list_path':'/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList-vti_camerons_annotations-split_peaks-417x286-rescale.csv',
        'file_list_all_peaks_path':'/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList-vti_camerons_annotations-split_peaks-417x286-rescale-all_peaks.csv',
        'batch_size':hyperparameters['batch_size'],
        'shuffle':True,
        'drop_last':False,
        'normalize_mode':'self',
        'target_type':'peak_array',
    
        ## Test settings     
        'train':True,
        'predict':True,
        'test_vti_from_all_tracings':True,
        'force_positive_velocity':hyperparameters['force_positive_velocity'],
        'plot_tracing':False,
    }
    
    loss = run_from_config(arg_dict=args)
    
    # returns the loss on validation set
    return loss

## Define the search space.
hyperparameter_space = {
    ## Force predicted velocities to be non-negative at test-time.
    # This was never been done before.
    'force_positive_velocity':hyperopt.hp.choice('force_positive_velocity', [True, False]),

    ## Batch size
    # Current best is 48. Seemed to always give good results in manual tuning.
    'batch_size':hyperopt.hp.choice('batch_size', [1,2,4,8,16,24,32,48,64]),
    
    ## Learning rate
    # Current best is 3e-4 = exp(-8.1)
    # Try log-normal: hp.qlognormal(label, mu, sigma, q)
    'learning_rate':hyperopt.hp.qlognormal('learning_rate', -8, 1, 1e-5),

    ## Weight decay used in Adam and AdamW optimizers
    # Current best is 0.008
    # Try uniform: hp.quniform(label, low, high, q)
    #'weight_decay':hyperopt.hp.quniform('weight_decay', 0.001, 0.1, 0.001),
    'weight_decay':hyperopt.hp.qlognormal('weight_decay', -9, 2, 1e-5),

    ## Training noise
    # Current best is 10.
    'train_noise':hyperopt.hp.quniform('train_noise', 0, 50, 0.1)
}


## Store the results of every iteration.
#bayes_trials = hyperopt.Trials()
trials_out_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/doppler/hyperopt_trials-loss_vti_mse-learning_rate-weight_decay-train_noise-batch_size-force_positive_velocity.pkl'
max_evaluations = 2000

# Optimize
best = hyperopt.fmin(fn=objective, space=hyperparameter_space, algo=hyperopt.tpe.suggest, max_evals=max_evaluations, trials_save_file=trials_out_path) # Autosave and autoload trial progress.

print(best)
