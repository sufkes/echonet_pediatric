#!/usr/bin/env python3

import os, sys
import glob
import pandas as pd
from matplotlib import pyplot as plt

#loss_paths = glob.glob(os.path.join(os.path.dirname(__file__), '../../runs/') + '*/loss.csv')
#loss_paths.sort()

log_paths = glob.glob(os.path.join(os.path.dirname(__file__), '../../runs/') + '*/log.csv')
log_paths.sort()

for log_path in log_paths:
    loss_df = pd.DataFrame(columns=['train', 'val', 'test'])
    loss_df.index.name = 'epoch'

    with open(log_path) as f:
        lines = f.readlines()

    lines = [l for l in lines if l.count(',') >= 8]
    for line in lines:
        epoch = int(line.split(',')[0])
        split = str(line.split(',')[1])
        loss = float(line.split(',')[2])
        loss_df.loc[epoch, split] = loss

    x = loss_df.index
    train = loss_df['train']
    val = loss_df['val']
    test = loss_df['test']

    plot_out_dir = os.path.dirname(log_path)
    plot_out_name = 'loss.png'
    plot_out_path = os.path.join(plot_out_dir, plot_out_name)
    
    plt.figure()
    plt.plot(x, train, '-ok', color='blue', label='training')
    plt.plot(x, val, '-ok', color='red', label='validation')
    plt.plot(x, test, '-ok', color='black', label='test')
    plt.legend()
    plt.savefig(plot_out_path)

    print('Saved:', plot_out_path)

    # Save the loss dataframe as csv for convenience.
    df_out_dir = os.path.dirname(log_path)
    df_out_name = 'loss.csv'
    df_out_path = os.path.join(df_out_dir, df_out_name)

    loss_df.to_csv(df_out_path, index=True)



