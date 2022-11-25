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
        "split_col": "split_LVEF",
        "output": f"/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/hyperopt-lvef-20220403-num_epochs_25/hyperopt-lvef-batch_size_{hyperparameters['batch_size']}-lr_{hyperparameters['lr']}-weight_decay_{hyperparameters['weight_decay']}",
        "freeze_n_conv_layers": 2,
        "set_fc_bias": True,
        "tasks": "LVEF_4Cand2C",
        "run_test": True,
        "run_train": True,
        "load_model_weights_only": True,
        "start_checkpoint_path": "/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/checkpoints/r2plus1d_18_32_2_pretrained.pt",
        "file_list_path": "/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/including_neonates_20220329/4c/FileList-mask_no-flip_yes-color_no.csv",
        "load_tracings": False,
        "file_path_col": "FilePath",
        "subject_name_col": "Subject",
        "num_workers": 0,
        "modelname": "r2plus1d_18",
        "frames": 32,
        "period": 2,
        "pretrained": True,
        "num_epochs": 25,
        "batch_size": hyperparameters["batch_size"],
        "optimizer_name": 'Adam',
        "lr": hyperparameters["lr"],
        "weight_decay": hyperparameters["weight_decay"],
        "momentum": 0.9, # not used in AdamW
        "lr_step_period": None,
        "return_value": "trained_model_val_loss_test_time_augmented",
        "note":"hyperopt on LVEF model with 25 epochs"
    }
    
    loss = run_from_config(arg_dict=args)
    
    return loss

hyperparameter_space = {
    ## Batch size
    # Formerly 20
    'batch_size':hyperopt.hp.choice('batch_size', list(range(2,26,2))),
    
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
trials_out_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/hyperopt-lvef-20220403-batch_size-lr-weight_decay-num_epochs_25.pkl'
max_evaluations = 1000

# Optimize
#best = hyperopt.fmin(fn=objective, space=hyperparameter_space, algo=hyperopt.tpe.suggest, max_evals=max_evaluations, trials=bayes_trials)
best = hyperopt.fmin(fn=objective, space=hyperparameter_space, algo=hyperopt.tpe.suggest, max_evals=max_evaluations, trials_save_file=trials_out_path) # Autosave and autoload trial progress.

print(best)
