import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from visualisations_utils_wandb_api import (
    download_runs,
    process_max_fields,
    process_runs,
    group_process_runs
    )

import os
import neatplot
neatplot.set_style('notex')


# Constants and configurations
ENTITY = 'robust-rl-project'
PROJECT = 'group-robust-dpo-neurips'
SETTINGS = {
    'goqa': ['goqa_2tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer'],
    'goqma': ['goqma_2tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer']
}
ALGORITHMS = {
    'rdpo': 'Robust DPO',
    'dpo': 'DPO',
    'sft': 'SFT',
    'base': 'Base_Model'
}

def get_setting_details(setting_key: str):
    if 'all' in setting_key:
        pass
    group_list = SETTINGS[setting_key]
    #weights_array = np.array(group_list[0].split('weights[')[-1].split(']')[0].split(','), dtype=float)
    #pref_data_num = group_list[0].split('pref_data_num')[1].split('weights')[0]
    return group_list#, weights_array, pref_data_num

def create_filter_dicts(groups: list[str],n_epochs: int):
    base_filter_dpo = {
        'config.loss.name': 'dpo',
        'State': 'finished',
        'config.n_epochs': n_epochs
    }
    base_filter_rdpo = {
        'State': 'finished',
        'config.loss.name': 'rdpo',
        'config.n_epochs': n_epochs
    }
    base_filter_sft = {
        'State': 'finished',
        'config.loss.name':'sft',
        'config.n_epochs': 1
    }
    base_filter_gemma_base = {
        'State': 'finished',
        'config.loss.name':'base'
    }

    if len(groups)==1: # IPO
        rdpo_filter = {**base_filter_rdpo, 'group': groups[0], 'config.loss.importance_sampling': False, 'config.loss.step_size': 0.1 }
        dpo_filter = {**base_filter_dpo, 'group': groups[0]}
        sft_filter = {**base_filter_sft, 'group': groups[0]}
        gemma_base_filter={**base_filter_gemma_base, 'group': groups[0]}
        return [rdpo_filter, dpo_filter, gemma_base_filter]

    filters = []
    for group in groups:
        filter = {
            **base_filter_dpo, 
            'group': group, 
            'config.ipo_grad_type': 'justdpo',
            'config.dpo_type': 'dpo' if 'dpo' in group else 'rdpo', 
            'config.importance_sampling': 'imp' in group,
            'config.importance_sampling_weights': {'$nin': ['0.5,0.5']}, 
            'config.use_theory': False
        }
        filters.append(filter)
    return filters

def determine_algorithm(filters_dict):
    if filters_dict['config.ipo_grad_type'] == 'linear': # IPO
        if not filters_dict['config.importance_sampling']:
            return 'RIPO Theory' if filters_dict['config.use_theory'] else 'RIPO Practice'
        if filters_dict['config.importance_sampling_weights'] == {'$in': ['0.5,0.5']}:
            return 'IPO'
        return 'IPO Importance Sampling'
    
    if filters_dict['config.importance_sampling'] == True:
        return 'DPO Importance Sampling'
    if filters_dict['config.dpo_type'] == 'dpo':
        return 'DPO'
    return 'RDPO'

def prepare_metric_data(filters_dicts, metrics,all_avg_metrics_at_iterations,all_sem_metrics_at_iterations,metric_titles):
    metric_values = []
    metric_sem = []
    labels = []
    for metric_name in metrics:
        for i,filters_dict in enumerate(filters_dicts):
            algo = determine_algorithm(filters_dict)
            avg = all_avg_metrics_at_iterations[metric_name][i]
            sem = all_sem_metrics_at_iterations[metric_name][i]
            #name=metric_titles[metric_name]
            metric_values.append(avg)
            metric_sem.append(sem)
            labels.append(algo)
    return metric_values, metric_sem, labels

def plot_metric_with_error_bands(iteration_index, metric_values, metric_sem, labels, plot_title, subfolder_path, file_name,metric, colors=None, extend=False):
    plt.figure(figsize=(12, 6))
    #for i, (avg, sem) in enumerate(zip(metric_values, metric_sem)):
    for avg, sem, label in zip(metric_values, metric_sem, labels):
        if extend and len(avg) != len(iteration_index):
            avg = np.append(avg, [avg[-1]] * (len(iteration_index) - len(avg)))
            sem = np.append(sem, [sem[-1]] * (len(iteration_index) - len(sem)))
        #color = colors[i] if colors else None
        plt.plot(iteration_index, avg, label=label)
        plt.fill_between(iteration_index, avg - sem, avg + sem, alpha=0.2)
    plt.title(plot_title,fontsize=40)
    plt.xlabel('Iterations',fontsize=40)
    plt.ylabel('Value',fontsize=40)
    plt.legend(fontsize=40)
    neatplot.save_figure(f'{subfolder_path}/{file_name}')
    plt.close()

def plot_metric_bars(metric_config, filters_dicts, subfolder_path, all_avg_metrics_at_iterations, all_sem_metrics_at_iterations):
    plt.figure(figsize=(12, 6))
    print(metric_config)

    color_map = {
        'base': 'blue',   # Blue for 'base'
        'sft': 'green',   # Green for 'sft'
        'rdpo': 'red',    # Red for 'rdpo'
        'dpo': 'purple'   # Purple for 'dpo'
    }

    # Define bar properties
    num_groups = 2
    num_bars_per_group = 4  # Adjust as needed per group
    bar_width = 0.2
    group_width = num_bars_per_group * bar_width + 0.1  # Adjust spacing between groups
    for i, filters_dict in enumerate(filters_dicts):
        algo = filters_dict['config.loss.name']
        # Initialize lists to store metrics for plotting
        groups = [[], []]  # Two groups for each type of metric

        # Gather data for each metric
        for metric in metric_config['metrics']:
            index = int(metric.split('_')[-1])  # This assumes metrics end in '_0' or '_1'
            category = ''
            if 'logps_accuracies_eval' in metric:
                category = 'base'
            elif 'logps_ref_eval/accuracies' in metric:
                category = 'sft'
            elif 'logps_pol_eval/accuracies' in metric:
                if 'rdpo' in algo:
                    category = 'rdpo' 
                    category_nan = "dpo"
                else:
                    category = 'dpo' 
                    category_nan = "rdpo"
                groups[index].append((category_nan,np.nan,np.nan))
            # Collect data
            avg = all_avg_metrics_at_iterations[metric][i][-1]
            sem = all_sem_metrics_at_iterations[metric][i][-1]
            groups[index].append((category, avg, sem))

        # Sort groups by predefined order if necessary
        order = ['base', 'sft', 'dpo', 'rdpo']
        groups = [sorted(g, key=lambda x: order.index(x[0])) for g in groups]
        print(groups)
        # Plot each group
        for group_index, group_data in enumerate(groups):
            for bar_index, data in enumerate(group_data):
                print(data)
                position = group_index * group_width + bar_index * bar_width
                plt.bar(position, height=data[1], yerr=data[2], width=bar_width, capsize=5,color=color_map[data[0]], alpha=0.7, label=f'{data[0]}' if i == 0 and group_index == 0 else "")

    plt.xticks([group_width/2 + i * group_width for i in range(num_groups)], ['Group 0', 'Group 1'], fontsize=20)
    plt.title(metric_config['title'],fontsize=40)
    plt.ylabel('Value',fontsize=40)
    plt.legend(fontsize=30)
    neatplot.save_figure(f'{subfolder_path}/{metric_config["file_suffix"]}')
    plt.close()

def main():
    setting = 'goqma'  # convention X_Y_Z: X={'even','uneven'}, Y={'balanced','imbalanced'}, Z={'dpo','ipo','all'}
    n_epochs=1
    groups= get_setting_details(setting)
    filters_dicts = create_filter_dicts(groups,n_epochs)
    
    #metrics_to_collect = ['grad_norm', 'train_loss', 'reward_err_1', 'reward_err_2', 'reward_param_1', 'reward_param_2', 'reward_param_3', 'reward_param_4','group_weight_1','group_weight_2','val_loss','train_group_loss_1','train_group_loss_2','val_group_loss_1','val_group_loss_2','hist_group_loss_1','hist_group_loss_2','max_val_grp_loss','max_train_grp_loss','max_reward_err','max_kl_dist']
    metrics_to_collect = ['logps_accuracies_eval_0','logps_accuracies_eval_1','logps_pol_eval/accuracies_0','logps_pol_eval/accuracies_1','logps_ref_eval/accuracies_0','logps_ref_eval/accuracies_1']
    all_metrics_history = {metric: [] for metric in metrics_to_collect}

    all_runs=[]
    # Loop through each filters_dict value
    for filters_dict in filters_dicts:
        # Download runs for the current filters_dict
        runs = download_runs(ENTITY, PROJECT, filters_dict)
        all_runs.append(runs)
        print(len(runs))
        metrics_history = {}

        for metric in metrics_to_collect:
            metrics_history[metric] = process_runs(runs, field=metric, time_field='_step')

        # Accumulate metrics data for each configuration
        for metric in metrics_to_collect:
            all_metrics_history[metric].append(metrics_history[metric])
    #print(all_metrics_history)
    iteration_len=0
    iteration_index=0
    for runs in all_runs:
        for run in runs:
            iteration_index_1=run[['_step']].dropna().values.ravel()
            #print(iteration_index_1)
            if len(iteration_index_1)>iteration_len:
                iteration_len=len(iteration_index_1)
                iteration_index=iteration_index_1


    base_folder = f'wandb-plots-gemma/{len(filters_dicts)}_setting_{setting}'
    os.makedirs(base_folder, exist_ok=True)
    subfolder_name = f"{filters_dicts[0]['config.loss.name']}{len(filters_dicts)}"
    subfolder_path = os.path.join(base_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    all_avg_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}
    all_sem_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}

    for i, filters_dict in enumerate(filters_dicts):
        for metric in metrics_to_collect:
            values_matrix = all_metrics_history[metric][i]
            #print('VAL MATRIX: ', values_matrix[0:2])
            #print(values_matrix)
            avg_values = np.mean(values_matrix, axis=0)
            sem_values = sem(all_metrics_history[metric][i], axis=0)
            all_avg_metrics_at_iterations[metric].append(avg_values.ravel())
            all_sem_metrics_at_iterations[metric].append(sem_values.ravel())

    #Plotting configurations
    plot_configs = [
        ('logps_accuracies_eval_0','logps_accuracies_eval_1')
    ]
    titles_dict = {
        'logps_accuracies': 'Log Likelihood Accuracies (Chosen vs Rejected)'
    }
    
    metrics_titles = {
        'logps_accuracies_eval_0' : 'Log Likelihood Accuracies Group-0',
        'logps_accuracies_eval_1' : 'Log Likelihood Accuracies Group-1'
    }


    #for metrics in plot_configs:
        #values, sems, labels = prepare_metric_data(filters_dicts, metrics,all_avg_metrics_at_iterations,all_sem_metrics_at_iterations,metrics_titles)
        #metric_name = "_".join(metrics)
        #plot_metric_with_error_bands(iteration_index, values, sems, labels, f'{metric_name} over Iterations', subfolder_path, f"{titles_dict[metric_name]}", metric, extend=True)
    # Define a list of metric configurations for each plot
    metrics_configs = [
        {'metrics': [metric for metric in metrics_to_collect if 'logps' in metric], 'title': 'Log Likelihood Accuracies (Chosen vs Rejected)', 'file_suffix': 'log_accuracy'}
        #{'metrics': [metric for metric in metrics_to_collect if 'train_group_loss' in metric], 'title': 'Group Train Loss at the End', 'file_suffix': 'train_group_loss_bars'},
        #{'metrics': [metric for metric in metrics_to_collect if 'val_group_loss' in metric], 'title': 'Group Validation Loss at the End', 'file_suffix': 'val_group_loss_bars'},
        #{'metrics': [metric for metric in metrics_to_collect if 'max_reward_err' in metric], 'title': 'Max Reward Error at the End', 'file_suffix': 'max_reward_bars'},
        #{'metrics': [metric for metric in metrics_to_collect if 'max_train_grp_loss' in metric], 'title': 'Max Group Train Loss at the End', 'file_suffix': 'max_train_group_loss_bars'},
        #{'metrics': [metric for metric in metrics_to_collect if 'max_val_grp_loss' in metric], 'title': 'Max Group Validation Loss at the End', 'file_suffix': 'max_val_group_loss_bars'},
        #{'metrics': [metric for metric in metrics_to_collect if 'max_kl_dist' in metric], 'title': 'Max KL Distance at the End', 'file_suffix': 'max_kl_distance_bars'}
    ]
    #print(metrics_configs)
    # Loop through each configuration and plot
    for config in metrics_configs:
        plot_metric_bars(config, filters_dicts, subfolder_path,all_avg_metrics_at_iterations,all_sem_metrics_at_iterations)

if __name__ == "__main__":
    main()


   



