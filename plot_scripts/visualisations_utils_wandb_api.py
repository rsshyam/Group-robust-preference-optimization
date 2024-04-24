# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:01:59 2023

@author: William
"""

import wandb
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'
        
def download_runs(entity, project, filters, num_samples=1e6):
    """
    Download runs via the wandb API

    Parameters
    ----------
    entity : str
        wandb entity name.
    project : str
        wandb project name.
    filters : dict
        filter to select runs in a MongoDB query format.

    Returns
    -------
    runs : list<pd.DataFrame>
        list of the run history retrieved from wandb.
    """
    
    api = wandb.Api(timeout=30)  
    runs = api.runs(entity + "/" + project, filters=filters)
    #print(entity,project,filters)
    #runs_config = [run.config for run in runs]        
    runs_hist = [run.history(samples=num_samples) for run in runs]
    
    return runs_hist #runs_hist, runs_config,
            
def process_runs(runs, field, time_field='epoch', agg='mean'):
    
    processed_runs = list()
    
    for df_run in runs:
        #print(df_run.columns)
        #Download the run's history
        if field not in df_run.columns:
            continue
        df = df_run[[field, time_field]]
        #print(df[field][:-1])
        #print(df)
        subset=df.iloc[:-1][field]
        #print(subset,'subset')
        is_string_nan_mask = subset.apply(lambda x: isinstance(x, str) and x.lower() == 'nan')
        
        #print(is_string_nan_mask.any(),'stringnan')
        if is_string_nan_mask.any():
            df[field] = df[field].replace('NaN', pd.NA)
        # print(df.isna())
        #    print(f"Skipping run due to NaN values in field '{field}'")
            #print(df[field])
        #    continue
        _filter = df[field].isna()
        df_filtered = df[~_filter]
        #print(df_filtered)
        #Assert numerical typing:
        df_filtered.loc[:, field] = df_filtered[field].astype(np.float64)
        
        df_filtered = df_filtered.groupby(time_field).agg({field:agg})
                       
        processed_runs.append(df_filtered)
        
    return processed_runs
            
def process_max_fields(runs, fields, maximum=True, time_field='epoch', x_percent_rmv=None):
    
    processed_runs = list()
    
    def remove_x_percent(series):
        pass
            
    for df_run in runs:
        
        run_results = list()
                
        for i, field in enumerate(fields):
                        
            df = df_run[[field, time_field]]
            _filter = df[time_field].isnull()
            df_filtered = df[~_filter]
            
            if x_percent_rmv is None:
                df_filtered = df_filtered.groupby(time_field).agg({field:'mean'})
            else:
                df_filtered = df_filtered.groupby(time_field).agg({field:remove_x_percent})


            run_results.append(df_filtered)
            
        df_concat = pd.concat(run_results, axis=1)
        
        if maximum:
            run_results = df_concat.max(axis=1)
        else:
            run_results = df_concat.min(axis=1)
        
        processed_runs.append(run_results)
        
    return processed_runs

    
def group_process_runs(processed_runs, runs):
    
    assert len(processed_runs) == len(runs),\
        'processed runs must be the same length as runs'
                
    #Stack together processed runs:
    df = pd.concat(processed_runs, axis=1)
    
    #Calculate the mean and std
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    
    return pd.concat([mean, std], axis=1)
         
        
def process_and_plot_max_grp_runs(fig, axs, runs, fields):
    pass

def process_and_plot_grp_runs(fig, axs, runs, fields):
    pass
    

    
    
    

    
#%% Test setup: plot runs from CINIC10

if __name__ == '__main__':

    entity='william_bankes'
    project='RobustRHO-CINICFinalResults'
    filters_mw = {'config.logger/wandb/tags':['CINIC10', 'robust']}
    mw_runs = download_runs(entity=entity, project=project, filters=filters_mw)
    
    filters_rho = {'config.logger/wandb/tags':['CINIC10', 'reducible']}
    rho_runs = download_runs(entity=entity, project=project, filters=filters_rho)

    fig, axs = plt.subplots()
    axs.set_title('Worst Class Test Accuracy')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Worst Class Test Accuracy')
    axs.set_ylim([0,1])
    
    fields = [f'class_{c}_val_acc_epoch' for c in range(10)]
    processed = process_max_fields(mw_runs, fields, maximum=False, time_field='epoch')
    grp_processed = group_process_runs(processed, mw_runs)
    
    mean = grp_processed[0]
    std  = grp_processed[1]
    axs.plot(grp_processed.index, mean, label='ReDuCe-Loss', color='tab:blue')
    axs.fill_between(grp_processed.index, mean-std, mean+std, alpha=0.2, color='tab:blue')
    
    fields = [f'class_{c}_val_acc_epoch' for c in range(10)]
    processed = process_max_fields(rho_runs, fields, maximum=False, time_field='epoch')
    grp_processed = group_process_runs(processed, rho_runs)
    
    mean = grp_processed[0]
    std  = grp_processed[1]
    axs.plot(grp_processed.index, mean, label='ReDuCe-Loss', color='tab:orange')
    axs.fill_between(grp_processed.index, mean-std, mean+std, alpha=0.2, color='tab:orange')