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
from collections import defaultdict
import os
import neatplot
import re
neatplot.set_style()


# Constants and configurations
ENTITY = 'entity-name'
PROJECT = 'project-name'
dataset_group='goqa'

#group=f'{dataset_group}_{group_indices}'+f'tr_frac{config.train_frac}'+f'{config.model.name_or_path}_spairs_{config.sep_pairs}_{config.trainer}',
            
SETTINGS = {
    'goqa': ['goqa_0_1tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer','goqa_2tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer'],#goqa_2tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer
    'goqma': ['goqma_2tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer'],
    'goqa_5':['goqa_0_1_2_3_4tr_frac0.8google/gemma-7b_spairs_False_GroupTrainer'],
    'goqa_5_gemma2b':['goqa_0_1_2_3_4tr_frac0.8google/gemma-2b_spairs_False_GroupTrainerEarlyStop']
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

def exponential_moving_average(data, alpha=0.9):
    """Compute the exponential moving average of a data series."""
    ema = np.zeros_like(data)
    ema[0] = data[0]  # Start the EMA with the first element of data
    for t in range(1, data.shape[0]):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

def create_filter_dicts(groups: list[str],n_epochs: int, base: bool=False, setting: str='dpo'):
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
    base_filter_ipo = {
        'config.loss.name': 'ipo',
        'State': 'finished',
        'config.n_epochs': n_epochs,
        'config.loss.beta': 0.01,
        'config.loss.label_smoothing': 0,
        'config.use_kfoldsplit': False,
        'config.optimizer': 'RMSprop',
        'config.model.archive': 'path-to-sft-file'
    }
    base_filter_ripo = {
        'State': 'finished',
        'config.loss.name': 'ripo',
        'config.n_epochs': n_epochs,
        'config.loss.beta': 0.01,
        'config.loss.label_smoothing': 0,
        'config.use_kfoldsplit': False,
        'config.optimizer': 'RMSprop',
        'config.model.archive': 'path-to-sft-file'
    }

    base_filter_ipo_adam = {
        'config.loss.name': 'ipo',
        'State': 'finished',
        'config.n_epochs': n_epochs,
        'config.loss.beta': 0.01,
        'config.loss.label_smoothing': 0,
        'config.use_kfoldsplit': False,
        'config.optimizer': 'AdamW',
        'config.model.archive': 'path-to-sft-file'
    }
    base_filter_ripo_adam = {
        'State': 'finished',
        'config.loss.name': 'ripo',
        'config.n_epochs': n_epochs,
        'config.loss.beta': 0.01,
        'config.loss.label_smoothing': 0,
        'config.use_kfoldsplit': False,
        'config.optimizer': 'AdamW',
        'config.model.archive': 'path-to-sft-file'
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

    #if len(groups)==1: # IPO
    if setting=='dpo':
        rdpo_filter = {**base_filter_rdpo, 'group': groups[0], 'config.loss.importance_sampling': False, 'config.loss.step_size': 0.05 }
        rdpo_filter_2 = {**base_filter_rdpo, 'group': groups[1], 'config.loss.importance_sampling': False, 'config.loss.step_size': 0.1 }
        dpo_filter = {**base_filter_dpo, 'group': groups[1]}
        sft_filter = {**base_filter_sft, 'group': groups[0]}
        gemma_base_filter={**base_filter_gemma_base, 'group': groups[0]}
        return [rdpo_filter, rdpo_filter_2, dpo_filter,gemma_base_filter] if base else [rdpo_filter, dpo_filter,  rdpo_filter_2]
    elif setting=='ipo':
        ripo_filter = {**base_filter_ripo, 'group': groups[0], 'config.loss.importance_sampling': False,'config.loss.divide_by_totalcount': True, 'config.loss.step_size': 0.000001} 
        ripo_filter_2 = {**base_filter_ripo, 'group': groups[0], 'config.loss.importance_sampling': False, 'config.loss.divide_by_totalcount': False, 'config.loss.step_size': 0.000005 }
        ipo_filter = {**base_filter_ipo, 'group': groups[0]}
        sft_filter = {**base_filter_sft, 'group': groups[0]}
        gemma_base_filter={**base_filter_gemma_base, 'group': groups[0]}
        return [ripo_filter, ripo_filter_2, ipo_filter,gemma_base_filter] if base else [ripo_filter, ipo_filter]
    elif setting=='ipo_earlystoploss':
        ripo_filter = {**base_filter_ripo, 'group': groups[0], 'config.loss.importance_sampling': False,'config.loss.divide_by_totalcount': True, 'config.loss.step_size': 0.0000001, 'config.patience_factor': 2, 'config.lr': 0.00002, 'config.scheduler_metric': 'loss','config.loss.step_factor': 0.5} 
        ripo_filter_2 = {**base_filter_ripo, 'group': groups[0], 'config.loss.importance_sampling': False, 'config.loss.divide_by_totalcount': True, 'config.loss.step_size': 0.000001, 'config.patience_factor': 2, 'config.lr': 0.0001, 'config.scheduler_metric': 'loss'}
        ipo_filter = {**base_filter_ipo, 'group': groups[0], 'config.patience_factor': 2, 'config.lr': 0.0001, 'config.scheduler_metric': 'loss', 'display_name': {"$regex":".*avg_fr_vald.*"}}
        ipo_filter_2 = {**base_filter_ipo, 'group': groups[0], 'config.patience_factor': 2, 'config.lr': 0.00001, 'config.scheduler_metric': 'loss', 'display_name': {"$regex":".*avg_fr_vald.*"}}
        sft_filter = {**base_filter_sft, 'group': groups[0]}
        gemma_base_filter={**base_filter_gemma_base, 'group': groups[0]}
        return [ripo_filter, ripo_filter_2, ipo_filter,gemma_base_filter] if base else [ripo_filter, ipo_filter_2]
    elif setting=='ipo_earlystoplossadam':
        ripo_filter = {**base_filter_ripo_adam, 'group': groups[0], 'config.loss.importance_sampling': False,'config.loss.divide_by_totalcount': True, 'config.loss.step_size': 0.0000005, 'config.patience_factor': 2, 'config.lr': 0.00006, 'config.scheduler_metric': 'loss','config.loss.step_factor': 0.5} 
        #ripo_filter_2 = {**base_filter_ripo, 'group': groups[0], 'config.loss.importance_sampling': False, 'config.loss.divide_by_totalcount': True, 'config.loss.step_size': 0.000001, 'config.patience_factor': 2, 'config.lr': 0.0001, 'config.scheduler_metric': 'loss'}
        #ipo_filter = {**base_filter_ipo, 'group': groups[0], 'config.patience_factor': 2, 'config.lr': 0.0001, 'config.scheduler_metric': 'loss', 'display_name': {"$regex":".*avg_fr_vald.*"}}
        ipo_filter_2 = {**base_filter_ipo_adam, 'group': groups[0], 'config.patience_factor': 2, 'config.lr': 0.00003, 'config.scheduler_metric': 'loss', 'display_name': {"$regex":".*avg_fr_vald.*"}}
        sft_filter = {**base_filter_sft, 'group': groups[0]}
        gemma_base_filter={**base_filter_gemma_base, 'group': groups[0]}
        evalonce_filter = {**base_filter_ipo_adam, 'config.eval_only_once': True}
        return [ripo_filter, ripo_filter_2, ipo_filter, gemma_base_filter, evalonce_filter] if base else [ripo_filter, ipo_filter_2, evalonce_filter]

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
    if filters_dict['config.loss.name'] in {'rdpo','ripo'}: # IPO
        algo=filters_dict['config.loss.name']
        if filters_dict['config.loss.importance_sampling'] == True:
            base_algo  = algo.split('r')[-1]
            return f'{base_algo} Importance Sampling'
        step_size=filters_dict['config.loss.step_size']
        lr=filters_dict['config.lr']
        divide_by_totalcount=filters_dict['config.loss.divide_by_totalcount']
        if divide_by_totalcount:
            if algo=='ripo':
                return 'GR-IPO'
            else:
                return 'GR-DPO'
            #return f'{algo}_step_{step_size}_lr_{lr}_theory'
        return f'{algo}_step_{step_size}_lr_{lr}_prac'
    
    if filters_dict['config.loss.name'] in {'dpo','ipo'}:
        algo=filters_dict['config.loss.name']
        lr=filters_dict['config.lr']
        #return f'{algo}_lr_{lr}'
        return 'IPO'
    return 'RDPO'

def prepare_metric_data(filters_dicts, metrics,all_avg_metrics_at_iterations,all_sem_metrics_at_iterations,metric_titles):
    metric_values = []
    metric_sem = []
    labels = []
    for metric_name in metrics:
        for i,filters_dict in enumerate(filters_dicts):
            algo = determine_algorithm(filters_dict)
           
            match = re.search(r"_(\d+)$", metric_name)
            if match:
                algo = algo + '_grp_' + match.group(1)
            #print(metric_name,algo,i)
            avg = all_avg_metrics_at_iterations[metric_name][i]
            sem = all_sem_metrics_at_iterations[metric_name][i]
            #name=metric_titles[metric_name]
            metric_values.append(avg)
            metric_sem.append(sem)
            labels.append(algo)
    return metric_values, metric_sem, labels

def plot_metric_with_error_bands(fig, axes, ax_index, iteration_index, metric_values, metric_sem, labels, plot_title, subfolder_path, file_name,metric, colors=None, eval_every=192, running_avg_window=None):

    titles={
        'worst_case_rewards_eval/accuracies over Iterations':'Min Rewards Accuracies (Test)',
        'worst_case_loss/eval over Iterations':'Max Group Loss (Test)',
        'worst_case_logps_pol_eval/accuracies over Iterations':'Min Log-Prob Accuracies (Test)',
        'rewards_train/weights over Iterations':'Group Weights (Train)',
    }

    lengths = []
    for i in range(len((metric_values))):
        lengths.append(len(metric_values[i]))
    end = min(lengths)
    #print('LENGTHS = ', lengths)
    #print('END = ', end)

    colors = {'IPO': 'tab:orange', 'GR-IPO': 'tab:blue'}

    #plt.figure(figsize=(12, 8))
    #for i, (avg, sem) in enumerate(zip(metric_values, metric_sem)):
    min_value = float('inf')
    max_value = float('-inf')    
    #print(metric_values,metric_sem,labels)
    for avg, sem, label in zip(metric_values, metric_sem, labels):

        if 'weights' not in plot_title:
            avg = avg[:end]
            sem = sem[:end]

        #if extend and len(avg) != len(iteration_index):
        #print(plot_title)
        match = re.search(r"_grp_(\d+)$", label)
        #print(label)
        if match or 'worst_case' in plot_title:
            #print(label)
            eval_every = eval_every  # replace with the actual value from config
            new_iteration_index = np.arange(0, iteration_index[-1] + 1, eval_every)
            #print(new_iteration_index,avg)
            #print(len(new_iteration_index),len(avg))
            new_iteration_index = np.append(new_iteration_index, iteration_index[-1])
            #if len(new_iteration_index) != len(avg):
            #    new_iteration_index = new_iteration_index[:len(avg)]
            #iteration_index = new_iteration_index

            if 'weights' in plot_title:
                avg = np.array(avg)
                avg = avg[np.arange(0, len(avg), int(eval_every/16))]

                sem = np.array(sem)
                sem = sem[np.arange(0, len(sem), int(eval_every/16))]

        else:
            new_iteration_index = iteration_index    
        #avg = np.append(avg, [avg[-1]] * (len(new_iteration_index) - len(avg)))
        #sem = np.append(sem, [sem[-1]] * (len(new_iteration_index) - len(sem)))
        #color = colors[i] if colors else None
        #print(avg,sem)

        if label=='GR-IPO':
            legend_label = r'$\textbf{' + label + '}$'
        elif 'GR-IPO_grp' in label:
            legend_label = f'GR-IPO Group {int(label[-1]) + 1}'
        else:
            legend_label = label

        if 'weights' in plot_title and 'grp' in label:
            if 'IPO' in label and 'GR-IPO' not in label:
                continue

            colors = {'GR-IPO_grp_0':'tab:brown', 'GR-IPO_grp_1':'tab:olive', 'GR-IPO_grp_2':'tab:cyan', 'GR-IPO_grp_3':'tab:purple', 'GR-IPO_grp_4':'tab:green'}
            c = label
        else:
            c = 'GR-IPO' if 'GR-IPO' in label else 'IPO'
        axes[ax_index].plot(new_iteration_index[:len(avg)], avg, label=legend_label, color=colors[c], linewidth=3)
            
        axes[ax_index].fill_between(new_iteration_index[:len(avg)], avg - sem, avg + sem, color=colors[c], alpha=0.2)

        axes[ax_index].grid(visible=True, linewidth=2)

        min_value = min(min_value, min(avg - sem))
        max_value = max(max_value, max(avg + sem))
        #print(min_value,max_value)

    if plot_title in titles.keys():
        axes[ax_index].set_title(titles[plot_title],fontdict={'fontsize':55})
    else:
        axes[ax_index].set_title(plot_title,fontdict={'fontsize':55})
    axes[ax_index].set_xlabel('Iterations',fontdict={'fontsize':55})
    axes[ax_index].set_ylabel('Value',fontdict={'fontsize':55})
    if 'weights' in plot_title:
        axes[ax_index].legend(fontsize=30,loc='center right')
    else:
        axes[ax_index].legend(fontsize=40)
    if 'accuracies' in plot_title:
        min_value = 0.6
        max_value = 0.8
    if 'loss/eval' in plot_title:
        min_value = 1500
    axes[ax_index].set_ylim(min_value, max_value)  # Set y-axis limits
    axes[ax_index].tick_params(axis='x',which='major',labelsize=40)
    axes[ax_index].tick_params(axis='y',which='major',labelsize=40)
    fig.tight_layout()  # Adjust spacing between subplots
    safe_title = file_name.replace('/', '-')
    ##neatplot.save_figure(f'{subfolder_path}/{safe_title}',ext_list='pdf')
    #plt.close()

def plot_metric_bars_dpo(fig, axes, ax_index, metric_config, filters_dicts, subfolder_path, all_avg_metrics_at_iterations, all_sem_metrics_at_iterations):

    titles={
        'worst_case_rewards_eval/accuracies':'Min Reward Accuracies (Test)',
        'worst_case_loss/eval':'Max Group Loss (Test)',
        'loss/eval':'Group Loss (Test)',
        'rewards_train/weights':'Converged Group Weights (Train)'
    }

    #plt.figure(figsize=(12, 8))
    all_positions = []
    all_algos = []
    for i, filters_dict in enumerate(filters_dicts):
        algo = determine_algorithm(filters_dict)
        metrics_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metric_config['metrics']]
        metrics_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metric_config['metrics']]
        
        bar_width = 0.1 if 'group_loss' in metric_config['metrics'][0] else 0.2
        offset = i * bar_width
        positions = np.arange(len(metrics_end_avg)) + offset
        all_positions.append(positions)

        def boldface(algo):
            if algo == 'GR-IPO':
                return r'$\textbf{' + algo + '}$'
            return algo

        all_algos.append([boldface(algo) for _ in range(len(metric_config['metrics']))])

        if algo=='GR-IPO':
            legend_label = r'$\textbf{' + algo + '}$'
        else:
            legend_label = f'{algo}'

        axes[ax_index].bar(positions, height=metrics_end_avg, yerr=metrics_end_sem, width=bar_width, capsize=5, alpha=0.7, label=legend_label)
        if 'worst_case' not in metric_config['title']:
            axes[ax_index].set_xticks(positions, [f"Group {i+1}" for i in range(len(metrics_end_avg))],fontdict={'fontsize':30})
            axes[ax_index].tick_params(axis='x',which='major',labelsize=35)
    if 'worst_case' in metric_config['title']:
        all_positions = np.array(all_positions).flatten()
        all_algos = np.array(all_algos).flatten()
        axes[ax_index].set_xticks(all_positions,all_algos,fontdict={'fontsize':50})
        axes[ax_index].tick_params(axis='x',which='major',labelsize=50)
    axes[ax_index].tick_params(axis='y',which='both',labelsize=50)
    if metric_config['title'] in titles.keys():
        axes[ax_index].set_title(titles[metric_config['title']],fontdict={'fontsize':55})
    else:
        axes[ax_index].set_title(metric_config['title'],fontdict={'fontsize':55})
    axes[ax_index].set_ylabel('Value',fontdict={'fontsize':55})
    if 'worst_case' not in metric_config['title']:
        if 'weights' in metric_config["title"]:
            loc = 'upper center'
        else:
            loc = 'lower right'
        axes[ax_index].legend(fontsize=40, loc=loc)
        axes[ax_index].set_xlabel('Groups',fontdict={'fontsize':55})
    else:
        axes[ax_index].set_xlabel('Methods',fontdict={'fontsize':55})
    #plt.legend(fontsize=25)
    safe_title = metric_config["title"].replace('/', '-')
    if 'worst_case_rewards_eval/accuracies' in metric_config['title']:
        axes[ax_index].set_ylim(0.68, 0.78)  # Set y-axis limits starting from 0.5
    if 'worst_case_loss/eval' in metric_config['title']:
        axes[ax_index].set_ylim(bottom=1500) #set y-axis lower limit to 1000
    elif 'loss/eval' in metric_config['title']:
        axes[ax_index].set_ylim(bottom=1000)

    fig.tight_layout()
        
    #neatplot.save_figure(f'{subfolder_path}/{safe_title}_bars',ext_list='pdf')
    #plt.close()

def plot_metric_bars_dpo_evalonce(fig, axes, ax_index, metrics, avg_metrics_at_iterations, sem_metrics_at_iterations):
    #plt.figure(figsize=(12, 8))

    #metrics = ['logps_pol_eval/accuracies_0', 'logps_pol_eval/accuracies_1', 'logps_pol_eval/accuracies_2', 'logps_pol_eval/accuracies_3', 'logps_pol_eval/accuracies_4']
    metrics_end_avg = [avg_metrics_at_iterations[metric][0][0] for metric in metrics]
    metrics_end_sem = [sem_metrics_at_iterations[metric][0] for metric in metrics]

    bar_width = 0.8        
    all_positions = []
    positions = np.arange(len(metrics_end_avg))
    all_positions.append(positions)

    print('VALS : ', metrics_end_avg, metrics_end_sem)

    colors = ['tab:brown', 'tab:olive', 'tab:cyan', 'tab:purple', 'tab:green']
    axes[ax_index].bar(positions, height=metrics_end_avg, yerr=metrics_end_sem, width=bar_width, capsize=5, alpha=0.7,color=colors)
    all_positions = np.array(all_positions).flatten()
    axes[ax_index].set_xticks(all_positions,[f'Group {i}' for i in range(1,len(metrics_end_avg)+1)],fontdict={'fontsize':40})

    axes[ax_index].tick_params(axis='x',which='major',labelsize=35)
    axes[ax_index].tick_params(axis='y',which='major',labelsize=35)
    
    axes[ax_index].set_title('Log-Prob Accuracies (Start)',fontdict={'fontsize':55})
    axes[ax_index].set_ylabel('Value',fontdict={'fontsize':55})
    axes[ax_index].set_xlabel('Groups',fontdict={'fontsize':50})

    #if 'worst_case_rewards_eval/accuracies' in metric_config['title']:
    axes[ax_index].set_ylim(bottom=0.5)  # Set y-axis limits starting from 0.5
        
    #neatplot.save_figure(f'{subfolder_path}/logps_pol_eval-accuracies_start_bars',ext_list='pdf')
    #plt.close()

def plot_metric_bars_dpo_opt_iter(metric_config, filters_dicts, subfolder_path, all_avg_metrics_at_iterations, all_sem_metrics_at_iterations,optimal_iteration_indices):
    #plt.figure(figsize=(12, 8))

    for i, filters_dict in enumerate(filters_dicts):
        algo = determine_algorithm(filters_dict)
        optimal_iteration_index=optimal_iteration_indices[i]
        #print(optimal_iteration_index)
        metrics_end_avg = [all_avg_metrics_at_iterations[metric][i][optimal_iteration_index] for metric in metric_config['metrics']]
        metrics_end_sem = [all_sem_metrics_at_iterations[metric][i][optimal_iteration_index] for metric in metric_config['metrics']]
        
        bar_width = 0.1 if 'group_loss' in metric_config['metrics'][0] else 0.2
        offset = i * bar_width
        positions = np.arange(len(metrics_end_avg)) + offset
        
        plt.bar(positions, height=metrics_end_avg, yerr=metrics_end_sem, width=bar_width, capsize=5, alpha=0.7, label=f'{algo}')
        plt.xticks(positions, [f"Group {i+1}" for i in range(len(metrics_end_avg))],fontsize=40)

    plt.title(metric_config['title'],fontsize=40)
    plt.ylabel('Value',fontsize=40)
    plt.legend(fontsize=40)
    safe_title = metric_config["title"].replace('/', '-')
    if 'accuracies' in safe_title:
        plt.ylim(0.5, 1)  # Set y-axis limits starting from 0.5
    if 'loss_eval' in safe_title:
        plt.ylim(bottom=1000) #set y-axis lower limit to 1000
        
    #neatplot.save_figure(f'{subfolder_path}/{safe_title}_bars')
    #plt.close()
    # Define bar properties

def plot_metric_bars(metric_config, filters_dicts, subfolder_path, all_avg_metrics_at_iterations, all_sem_metrics_at_iterations):
    #plt.figure(figsize=(12, 6))
    #print(metric_config)

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
            else:
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
        #print(groups)
        groups = [sorted(g, key=lambda x: order.index(x[0])) for g in groups]
        #print(groups)
        # Plot each group
        for group_index, group_data in enumerate(groups):
            for bar_index, data in enumerate(group_data):
                #print(data)
                position = group_index * group_width + bar_index * bar_width
                plt.bar(position, height=data[1], yerr=data[2], width=bar_width, capsize=5,color=color_map[data[0]], alpha=0.7, label=f'{data[0]}' if i == 0 and group_index == 0 else "")

    plt.xticks([group_width/2 + i * group_width for i in range(num_groups)], ['Group 0', 'Group 1'], fontsize=20)
    plt.title(metric_config['title'],fontsize=40)
    plt.ylabel('Value',fontsize=40)
    plt.legend(fontsize=30)
    #neatplot.save_figure(f'{subfolder_path}/{metric_config["file_suffix"]}')
    #plt.close()

def generate_metrics(base_name, count, mode='eval', separator='_'):
    metrics = []
    if '/' in base_name:
        # Split the base name on the slash
        parts = base_name.split('/')
        # Reconstruct the base name with the mode inserted before the slash
        modified_base_name = f'{parts[0]}{separator}{mode}/{parts[1]}'
    else:
        # If no slash, proceed normally
        modified_base_name = f'{base_name}{separator}{mode}'

    # Generate metric names based on the modified base name
    for i in range(count):
        metrics.append(f'{modified_base_name}_{i}')
    metrics.append(f'worst_case_{modified_base_name}')
    return metrics
def pad_with_last_value(seq, max_length):
    last_value = seq[-1]
    padding = [last_value] * (max_length - len(seq))
    return seq + padding

def pad_series_with_last_value(series, max_length):
    last_value = series.iloc[-1]
    padding = [last_value] * (max_length - len(series))
    padded_series = series.tolist() + padding
    return padded_series

def main():
    setting = 'goqa_5_gemma2b'  # convention X_Y_Z: X={'even','uneven'}, Y={'balanced','imbalanced'}, Z={'dpo','ipo','all'}
    n_epochs=30
    group_count=5
    groups= get_setting_details(setting)
    algo_setting='ipo_earlystoplossadam'
    eval_every=960
    filters_dicts = create_filter_dicts(groups,n_epochs,setting=algo_setting)
    smoothing_alpha = 1
    
    #metrics_to_collect = ['grad_norm', 'train_loss', 'reward_err_1', 'reward_err_2', 'reward_param_1', 'reward_param_2', 'reward_param_3', 'reward_param_4','group_weight_1','group_weight_2','val_loss','train_group_loss_1','train_group_loss_2','val_group_loss_1','val_group_loss_2','hist_group_loss_1','hist_group_loss_2','max_val_grp_loss','max_train_grp_loss','max_reward_err','max_kl_dist']
    #metrics_to_collect = ['logps_accuracies_eval_0','logps_accuracies_eval_1','logps_pol_eval/accuracies_0','logps_pol_eval/accuracies_1','logps_ref_eval/accuracies_0','logps_ref_eval/accuracies_1','loss/train_0','loss/train_1','loss/eval_0','loss/eval_1']
    

    # Define the configuration for each metric group
    metric_configurations = [
        {'base_name': 'logps/chosen', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'logps/rejected', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'logps/chosen', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'logps/rejected', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'logps_pol/accuracies', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'logps_pol/accuracies', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'logps_ref/accuracies', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'logps_ref/accuracies', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'rewards/chosen', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'rewards/rejected', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'rewards/chosen', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'rewards/rejected', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'rewards/accuracies', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'rewards/accuracies', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'rewards/margins', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'rewards/margins', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'loss', 'count': group_count, 'mode': 'eval', 'separator': '/'},
        {'base_name': 'loss', 'count': group_count, 'mode': 'train', 'separator': '/'},
        {'base_name': 'logps/chosen', 'count': group_count, 'mode': 'vald', 'separator': '_'},
        {'base_name': 'logps/rejected', 'count': group_count, 'mode': 'vald', 'separator': '_'},
        {'base_name': 'logps_pol/accuracies', 'count': group_count, 'mode': 'vald', 'separator': '_'},
        {'base_name': 'rewards/chosen', 'count': group_count, 'mode': 'vald', 'separator': '_'},
        {'base_name': 'rewards/rejected', 'count': group_count, 'mode': 'vald', 'separator': '_'},
        {'base_name': 'rewards/accuracies', 'count': group_count, 'mode': 'vald', 'separator': '_'},
        {'base_name': 'rewards/margins', 'count': group_count, 'mode': 'vald', 'separator': '_'},
        {'base_name': 'loss', 'count': group_count, 'mode': 'vald', 'separator': '/'},
        {'base_name': 'rewards/weights', 'count': group_count, 'mode': 'train', 'separator': '_'},
    ]

    # Initialize an empty list to collect all generated metrics
    metrics_list = []

    # Generate metrics for each configuration and add them to the list
    for config in metric_configurations:
        generated_metrics = generate_metrics(
            base_name=config['base_name'],
            count=config['count'],
            mode=config['mode'],
            separator=config['separator']
        )
        metrics_list.extend([metric for metric in generated_metrics if metric != 'worst_case_rewards_train/weights'])

    # Print the full list of generated metrics
    
    #metrics_list.append('loss/train')
    metrics_to_collect=metrics_list
    print(metrics_to_collect)
        
    all_metrics_history = {metric: [] for metric in metrics_to_collect}
    evalonce_metrics_history = {metric: [0] for metric in metrics_to_collect}

    all_runs=[]
    processed_filters = []
    # Loop through each filters_dict value
    for filters_dict in filters_dicts:
        # Download runs for the current filters_dict
        runs = download_runs(ENTITY, PROJECT, filters_dict)
        if len(runs)>0:
            all_runs.append(runs)
            print(len(runs),filters_dict)
            metrics_history = {}

            for metric in metrics_to_collect:
                metrics_history[metric] = process_runs(runs, field=metric, time_field='_step')

            # Accumulate metrics data for each configuration
            for metric in metrics_to_collect:
                if 'config.eval_only_once' not in filters_dict:
                    all_metrics_history[metric].append(metrics_history[metric])
                else:
                    evalonce_metrics_history[metric] = metrics_history[metric]
            processed_filters.append(filters_dict)

    #print(all_metrics_history)
    filters_dict=processed_filters
    iteration_len=0
    iteration_index=0
    for runs in all_runs:
        for run in runs:
            iteration_index_1=run[['_step']].dropna().values.ravel()
            #print(iteration_index_1)
            if len(iteration_index_1)>iteration_len:
                iteration_len=len(iteration_index_1)
                iteration_index=iteration_index_1
    #print(iteration_index,iteration_len)
    
    base_folder = f'wandb-plots-gemma/{len(filters_dicts)}_setting_{setting}_{algo_setting}'
    os.makedirs(base_folder, exist_ok=True)
    subfolder_name = f"{filters_dicts[0]['config.loss.name']}{len(filters_dicts)}_{smoothing_alpha}_opt"
    subfolder_path = os.path.join(base_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    all_avg_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}
    all_sem_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}
    
    evalonce_avg_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}
    evalonce_sem_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}

    for i, filters_dict in enumerate(filters_dicts):
        #optimal_iteration_index=np.argmax(avg_values.squeeze())
        #print(optimal_iteration_index)
        #optimal_iteration_indices.append(optimal_iteration_index)
        for metric in metrics_to_collect:
            
            if 'config.eval_only_once' in filters_dict:            
                if 'logps_pol_train/accuracies_' not in metric:
                    all_avg_metrics_at_iterations[metric].append(0)
                    all_sem_metrics_at_iterations[metric].append(0)
                    continue

                hist = evalonce_metrics_history[metric]
                initial_logps = hist[0].iloc[0]

                evalonce_avg_metrics_at_iterations[metric].append(initial_logps)
                evalonce_sem_metrics_at_iterations[metric].append(0)

                continue
            
            #print(all_metrics_history[metric])
            values_matrix = all_metrics_history[metric][i]
            #print('VAL MATRIX: ', values_matrix[0:2])
            # Find the maximum length of the sequences
            #print(values_matrix)
            max_length = max(len(seq) for seq in values_matrix)

            # Pad the sequences with their last value to make them of equal length
            padded_values_matrix = [pad_series_with_last_value(seq.iloc[:,0], max_length) for seq in values_matrix]

            # Convert to a NumPy array
            values_matrix = np.array(padded_values_matrix)
            #values_matrix = np.array(values_matrix)
            #print(values_matrix.shape)

            avg_values = np.mean(values_matrix, axis=0)
            sem_values = sem(values_matrix, axis=0)
            sem_values[np.isnan(sem_values)] = 0
            all_avg_metrics_at_iterations[metric].append(avg_values.ravel())
            all_sem_metrics_at_iterations[metric].append(sem_values.ravel())
    #print(optimal_iteration_indices)
    #print(all_sem_metrics_at_iterations)
    # Create a default dictionary to hold the grouped metrics
    optimal_iteration_indices=None
    plot_configs = defaultdict(list)

    # Parse each metric and group by a derived key (ignoring the numeric suffix)
    for metric in metrics_to_collect:
        if 'worst_case' not in metric:
            key_parts = metric.rsplit('_', 1)
            key = key_parts[0]  # Split off the numeric suffix
            plot_configs[key].append(metric)  # Append metric to its group
        else:
            plot_configs[metric] = [metric]  # Create a separate key for each metric

    # Convert defaultdict to a regular dict
    plot_configs_dict = dict(plot_configs)
    
    #print(plot_configs_dict)
    titles_dict = {
        'logps_accuracies': 'Log Likelihood Accuracies (Chosen vs Rejected)',
        'loss/train': 'Group Train Loss',
        'loss/eval': 'Group Validation Loss'
    }
    
    metrics_titles = {
        'logps_accuracies_eval_0' : 'Log Likelihood Accuracies Group-0',
        'logps_accuracies_eval_1' : 'Log Likelihood Accuracies Group-1'
    }

    # specify here what metrics to plot as evolution plots and bar plot. Choose 1 of the 2 configs below:
    evolution_plot_metrics = [] 
    #evolution_plot_metrics = [['rewards_train/weights_0', 'rewards_train/weights_1', 'rewards_train/weights_2', 'rewards_train/weights_3', 'rewards_train/weights_4'], ['worst_case_logps_pol_eval/accuracies']]
    
    bar_plot_metrics = [['worst_case_loss/eval'], ['worst_case_rewards_eval/accuracies'], ['loss/eval_0', 'loss/eval_1', 'loss/eval_2', 'loss/eval_3', 'loss/eval_4']]
    #bar_plot_metrics = [['logps_pol_train/accuracies_0', 'logps_pol_train/accuracies_1', 'logps_pol_train/accuracies_2', 'logps_pol_train/accuracies_3', 'logps_pol_train/accuracies_4']]

    subplot_plotter(
        evolution_plot_metrics,
        bar_plot_metrics,
        filters_dicts,
        subfolder_path, 
        evalonce_avg_metrics_at_iterations,
        evalonce_sem_metrics_at_iterations,
        all_avg_metrics_at_iterations,
        all_sem_metrics_at_iterations,
        iteration_index,
        eval_every,
    )

def subplot_plotter(
        evolution_plot_metrics,
        bar_plot_metrics,
        filters_dicts,
        subfolder_path, 
        evalonce_avg_metrics_at_iterations,
        evalonce_sem_metrics_at_iterations,
        all_avg_metrics_at_iterations,
        all_sem_metrics_at_iterations,
        iteration_index,
        eval_every
    ):

    fig, axes = plt.subplots(1, len(evolution_plot_metrics)+len(bar_plot_metrics), figsize=(40,8))
    plt.subplots_adjust(wspace=4)

    ax_index = 0
    valid_filters_dicts = [filter_dict for filter_dict in filters_dicts if 'config.eval_only_once' not in filter_dict]
    
    for metric in evolution_plot_metrics:
        if 'worst_case' not in metric[0]:
            key_parts = metric[0].rsplit('_', 1)
            title = key_parts[0]  # Split off the numeric suffix
        else:
            title = metric[0]

        values, sems, labels = prepare_metric_data(valid_filters_dicts, metric, all_avg_metrics_at_iterations, all_sem_metrics_at_iterations, metric)
        plot_metric_with_error_bands(fig, axes, ax_index, iteration_index, values, sems, labels, f'{title} over Iterations', subfolder_path, title, metric[0], eval_every=eval_every)
        ax_index += 1

    for metric in bar_plot_metrics:
        if 'worst_case' not in metric[0]:
            key_parts = metric[0].rsplit('_', 1)
            title = key_parts[0]  # Split off the numeric suffix
        else:
            title = metric[0]

        if 'logps_pol_train/accuracies' in metric[0]:
            plot_metric_bars_dpo_evalonce(fig, axes, ax_index, metric, evalonce_avg_metrics_at_iterations, evalonce_sem_metrics_at_iterations)
        else:
            plot_metric_bars_dpo(fig, axes, ax_index, {'title': title, 'metrics': metric}, valid_filters_dicts, subfolder_path, all_avg_metrics_at_iterations, all_sem_metrics_at_iterations)
        ax_index += 1

    safe_title = title.replace('/', '-')
    neatplot.save_figure(f'{subfolder_path}/subplot_metrics_{safe_title}', ext_list='pdf')
    plt.close()

if __name__ == "__main__":
    main()            

