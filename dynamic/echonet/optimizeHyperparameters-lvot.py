#!/usr/bin/env python

import os
import sys

import hyperopt

from utils.video import run_from_config

## Define the objective function.
def objective(hyperparameters):
    print(hyperparameters)
    # train_and_evaluate() expects a bunch of keyword arguments. Replace those which I want to optimize with the values in params. Set others to fixed values.
    args = {
        "split_col": "split_all_random",
        "output": f"/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/hyperopt-optimizer_name_{hyperparameters['optimizer_name']}-batch_size_{hyperparameters['batch_size']}-lr_{hyperparameters['lr']}-weight_decay_{hyperparameters['weight_decay']}",
        "freeze_n_conv_layers": 2,
        "set_fc_bias": False,
        "tasks": "LVOT_px_trainoneframe",
        "run_test": False,
        "run_train": True,
        "load_model_weights_only": True,
        "start_checkpoint_path": "/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/checkpoints/r2plus1d_18_32_2_pretrained.pt",
        "file_list_path": "/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/lvot/FileList_lvot-augmented.csv",
        "load_tracings": False,
        "file_path_col": "FilePath",
        "subject_name_col": "Subject",
        "num_workers": 0,
        "modelname": "r2plus1d_18",
        "frames": 32,
        "period": 2,
        "pretrained": True,
        "num_epochs": 20,
        "batch_size": hyperparameters["batch_size"],
        "optimizer_name": hyperparameters["optimizer_name"],
        "lr": hyperparameters["lr"],
        "weight_decay": hyperparameters["weight_decay"],
        "momentum": 0.9, # not used in AdamW
        "lr_step_period": None,
        "note":"first trial hyperopt on LVOT model"
    }
    
    loss = run_from_config(arg_dict=args)
    
    # returns the loss on validation set
    return loss

hyperparameter_space = {
    ## Optimzer
    # Formerly SGD
    'optimizer_name':hyperopt.hp.choice('optimizer_name', ['Adam', 'AdamW']),

    ## Batch size
    # Formerly 20
    'batch_size':hyperopt.hp.choice('batch_size', [1,2,4,8,16,20,32,48]),
    
    ## Learning rate.
    # Formerly 1e-4 (with decay schedule)
    # log-normal: hp.qlognormal(label, mu, sigma, q)
    'lr':hyperopt.hp.qlognormal('learning_rate', -8, 1, 1e-5),

    ## Weight decay used in Adam and AdamW optimizers.
    # Formerly 1e-4
    'weight_decay':hyperopt.hp.qlognormal('weight_decay', -9, 2, 1e-5)
}


## Store the results of every iteration.
#bayes_trials = hyperopt.Trials()
trials_out_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/hyperopt_trials-lvot-optimizer_name-batch_size-lr-weight_decay.pkl'
max_evaluations = 200

# Optimize
#best = hyperopt.fmin(fn=objective, space=hyperparameter_space, algo=hyperopt.tpe.suggest, max_evals=max_evaluations, trials=bayes_trials)
best = hyperopt.fmin(fn=objective, space=hyperparameter_space, algo=hyperopt.tpe.suggest, max_evals=max_evaluations, trials_save_file=trials_out_path) # Autosave and autoload trial progress.

print(best)
