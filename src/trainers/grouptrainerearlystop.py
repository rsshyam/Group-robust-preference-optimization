import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import csv

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from src.preference_datasets import get_batch_iterator
from src.utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
from src.data_selection import DataSelector
from src.loss_utils import (
    preference_loss,
    _get_batch_logps,
    concatenated_inputs)

import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


from src.trainers.basictrainer import BasicTrainer



def get_loss_kwargs(loss_config: DictConfig):
    if 'dpo' in loss_config.name:
        loss_kwargs = {
                'beta': loss_config.beta,
                'reference_free': loss_config.get('reference_free', False),
                'label_smoothing': loss_config.get('label_smoothing', 0),
                'ipo': loss_config.name == 'ipo'
            }
    elif 'ipo' in loss_config.name:
        loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
    else:
        raise ValueError(f'unknown loss {loss_config.name}')
    return loss_kwargs

class GroupTrainerEarlyStop(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, data_selector: DataSelector = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        #robust aspects
        self.loss=[]
        self.group_idx=[]

        self.epoch_offset=1##to be changed
        self.n_groups=len(config.datasets)
        self.normalize_loss=False
        
        if config.use_kfoldsplit:
            self.split_idx=config.seed
        else:
            self.split_idx=None

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
            weighted=config.weighted_batches,
            sep_pairs=config.sep_pairs,
            group_handling=True,
            test_dataset=config.test_dataset,
            train_frac=config.train_frac,
            seed=self.seed
        )

        self.policy = policy
        self.reference_model = reference_model
        self.group_counts= get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs),mode='count_groups',split_idx=self.split_idx)
        self.total_count=sum(self.group_counts)
        if self.total_count>2000 and config.loss.name != 'sft':
            rank0_print('creating validation set for early stopping')
            self.early_stopping=True
        else:
            rank0_print('dataset too small for validation set of early stopping')
            self.early_stopping=False

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split=f'train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs),split_idx=self.split_idx)
        
        
        rank0_print(f'Loaded train data iterator')
        if config.loss.name in {'rdpo','ripo'}:
            self.set_adjustments_impsamp()
        #both functions need to be implemented
        if not self.early_stopping:
            self.prepare_eval_vald_iterator()
        else:
            self.prepare_eval_vald_iterator_earlystop()

        self.data_selector = data_selector


    def set_adjustments_impsamp(self):
         #loss_var
        if self.config.loss.adj is not None:
            # process generalization adjustment stuff
            adjustments = [float(self.config.loss.adj)]
            assert len(adjustments) in (1, self.n_groups)
            if len(adjustments)==1:
                adjustments = np.array(adjustments* self.n_groups)
            else:
                adjustments = np.array(adjustments)
            self.adj = torch.from_numpy(adjustments).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()
        self.group_counts=torch.LongTensor(self.group_counts).cuda()
        if self.config.loss.importance_sampling==False:
            self.adv_probs = torch.ones(self.n_groups).float().cuda()/self.n_groups
        else:
            if self.config.loss.imp_weights==False:
                if self.config.loss.dpowts==True:
                    self.adv_probs = 0.5*torch.ones(self.n_groups).cuda()
                    self.adv_probs = self.adv_probs.float()
                else:
                    self.adv_probs = torch.ones(self.n_groups).float().cuda()/self.group_counts
                    self.adv_probs = self.adv_probs/(self.adv_probs.sum()).float()
            else:
                self.adv_probs = torch.tensor(self.config.loss.imp_weights).float().cuda()
    
    def prepare_eval_vald_iterator(self):
        data_iterator_kwargs_eval={}
        for i in range(len(self.config.datasets)):
            rank0_print(self.config.datasets[i:i+1])
            data_iterator_kwargs_eval[i]=dict(
            names=self.config.datasets[i:i+1],
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            sft_mode=self.config.loss.name == 'sft',
            weighted=self.config.weighted_batches,
            sep_pairs=self.config.sep_pairs,
            group_handling=True,
            test_dataset=self.config.test_dataset
            )#separates datasets}
        self.traineval_iterator={}
        self.traineval_batches={}
        self.traingen_iterator={}
        self.traingen_batches={}
        if self.config.eval_train_data == True or self.config.eval_train_end==True: #add or
            if self.config.eval_train_full == True:#evaluate part of train data at the end to compare
                for i in range(len(data_iterator_kwargs_eval)):
                    ##for metrics
                    self.traineval_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='train', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                    rank0_print(f'Loaded Train-eval data iterator {i}')
                    self.traineval_batches[i] = list(self.traineval_iterator[i])

                    ###for sampling
                    self.traingen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='train_gen', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                    rank0_print(f'Loaded Train-gen data iterator {i}')
                    self.traingen_batches[i] = list(self.traingen_iterator[i])
                
            else:
                for i in range(len(data_iterator_kwargs_eval)):
                    ##for metrics
                    self.traineval_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='train', n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                    rank0_print(f'Loaded Train-eval data iterator {i}')
                    self.traineval_batches[i] = list(self.traineval_iterator[i])
                    #wandb.log({'train_eval_batches': len(self.traineval_batches[i])})

                    ###for sampling
                    self.traingen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='train_gen',  n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                    rank0_print(f'Loaded Train-gen data iterator {i}')
                    self.traingen_batches[i] = list(self.traingen_iterator[i])

        self.eval_iterator={}
        self.eval_batches={}
        self.gen_iterator={}
        self.gen_batches={}
        if self.config.eval_full==True:
            for i in range(len(data_iterator_kwargs_eval)):
                ###for metrics
                self.eval_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='test', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                self.eval_batches[i] = list(self.eval_iterator[i])
                rank0_print(f'Loaded {len(self.eval_batches[i])} eval batches of size {self.config.eval_batch_size} from {i}')

                ###for sampling
                self.gen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='test_gen', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                rank0_print(f'Loaded Test-gen data iterator {i}')
                self.gen_batches[i] = list(self.gen_iterator[i])

        else:
            for i in range(len(data_iterator_kwargs_eval)):
                ####for metrics
                self.eval_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='test', n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                self.eval_batches[i] = list(self.eval_iterator[i])
                rank0_print(f'Loaded {len(self.eval_batches[i])} eval batches of size {self.config.eval_batch_size} from 1')
                #wandb.log({'eval_batches': len(self.eval_batches[i])})

                ###for sampling
                self.gen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='test_gen',  n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                rank0_print(f'Loaded Test-gen data iterator {i}')
                self.gen_batches[i] = list(self.gen_iterator[i])


    def prepare_eval_vald_iterator_earlystop(self):
        data_iterator_kwargs_eval={}
        for i in range(len(self.config.datasets)):
            rank0_print(self.config.datasets[i:i+1])
            data_iterator_kwargs_eval[i]=dict(
            names=self.config.datasets[i:i+1],
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            sft_mode=self.config.loss.name == 'sft',
            weighted=self.config.weighted_batches,
            sep_pairs=self.config.sep_pairs,
            group_handling=True,
            test_dataset=self.config.test_dataset
            )#separates datasets}
        self.traineval_iterator={}
        self.traineval_batches={}
        self.traingen_iterator={}
        self.traingen_batches={}
        if self.config.eval_train_data == True or self.config.eval_train_end==True: #add or
            if self.config.eval_train_full == True:#evaluate part of train data at the end to compare
                for i in range(len(data_iterator_kwargs_eval)):
                    ##for metrics
                    self.traineval_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='train', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                    rank0_print(f'Loaded Train-eval data iterator {i}')
                    self.traineval_batches[i] = list(self.traineval_iterator[i])

                    ###for sampling
                    self.traingen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='train_gen', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                    rank0_print(f'Loaded Train-gen data iterator {i}')
                    self.traingen_batches[i] = list(self.traingen_iterator[i])
                
            else:
                for i in range(len(data_iterator_kwargs_eval)):
                    ##for metrics
                    self.traineval_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='train', n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                    rank0_print(f'Loaded Train-eval data iterator {i}')
                    self.traineval_batches[i] = list(self.traineval_iterator[i])
                    #wandb.log({'train_eval_batches': len(self.traineval_batches[i])})

                    ###for sampling
                    self.traingen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='train_gen',  n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                    rank0_print(f'Loaded Train-gen data iterator {i}')
                    self.traingen_batches[i] = list(self.traingen_iterator[i])

        self.eval_iterator={}
        self.eval_batches={}
        self.gen_iterator={}
        self.gen_batches={}
        if self.config.eval_full==True:
            for i in range(len(data_iterator_kwargs_eval)):
                ###for metrics
                self.eval_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='truetest', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                self.eval_batches[i] = list(self.eval_iterator[i])
                rank0_print(f'Loaded {len(self.eval_batches[i])} eval batches of size {self.config.eval_batch_size} from {i}')

                ###for sampling
                self.gen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='truetest_gen', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                rank0_print(f'Loaded Test-gen data iterator {i}')
                self.gen_batches[i] = list(self.gen_iterator[i])

        else:
            for i in range(len(data_iterator_kwargs_eval)):
                ####for metrics
                self.eval_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='truetest', n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                self.eval_batches[i] = list(self.eval_iterator[i])
                rank0_print(f'Loaded {len(self.eval_batches[i])} eval batches of size {self.config.eval_batch_size} from {i}')
                #wandb.log({'eval_batches': len(self.eval_batches[i])})

                ###for sampling
                self.gen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='truetest_gen',  n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                rank0_print(f'Loaded Test-gen data iterator {i}')
                self.gen_batches[i] = list(self.gen_iterator[i])

        self.vald_iterator={}
        self.vald_batches={}
        self.valdgen_iterator={}
        self.valdgen_batches={}
        if self.config.eval_full==True:
            for i in range(len(data_iterator_kwargs_eval)):
                ###for metrics
                self.vald_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='valtest', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                self.vald_batches[i] = list(self.vald_iterator[i])
                rank0_print(f'Loaded {len(self.vald_batches[i])} validation batches of size {self.config.eval_batch_size} from {i}')

                ###for sampling
                self.valdgen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='valtest_gen', n_epochs=self.config.n_epochs, n_examples=self.config.n_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                rank0_print(f'Loaded validation-gen data iterator {i}')
                self.valdgen_batches[i] = list(self.valdgen_iterator[i])

        else:
            for i in range(len(data_iterator_kwargs_eval)):
                ####for metrics
                self.vald_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='valtest', n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                self.vald_batches[i] = list(self.vald_iterator[i])
                rank0_print(f'Loaded {len(self.eval_batches[i])} validation batches of size {self.config.eval_batch_size} from {i}')
                #wandb.log({'eval_batches': len(self.eval_batches[i])})

                ###for sampling
                self.valdgen_iterator[i] = get_batch_iterator(**data_iterator_kwargs_eval[i], split='valtest_gen',  n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size, silent=self.rank != 0, cache_dir=get_local_dir(self.config.local_dirs),split_idx=self.split_idx)
                rank0_print(f'Loaded Test-gen data iterator {i}')
                self.valdgen_batches[i] = list(self.valdgen_iterator[i])



    
    def get_group_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute loss and metrics for the given batch of inputs, supporting both individual and group-based metrics."""
        metrics = {}
        train_test = 'train' if train else 'eval'

        # Compute loss based on the loss configuration
        if loss_config.name in {'dpo', 'ipo', 'rdpo', 'ripo'}:
            # Common forward pass for policy and reference models
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)
            loss_kwargs=get_loss_kwargs(loss_config)
            losses, chosen_rewards, rejected_rewards = preference_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            #actual_loss,metrics=self.get_group_metrics(losses,chosen_rewards,rejected_rewards,reward_accuracies,loss_config,train_test, group_idx=batch['group'])
            group_idx=batch['group']
            group_acc, group_count = self.compute_group_sum_metric(reward_accuracies, group_idx)
            if loss_config.name in {'dpo', 'ipo'}:
                actual_loss=losses.mean()
                weights=torch.ones(self.n_groups).float().cuda()/self.n_groups
            elif loss_config.name in {'rdpo', 'ripo'}:
                group_loss, group_count = self.compute_group_metric(losses, group_idx, self.config.loss.divide_by_totalcount)
                actual_loss, weights = self.compute_robust_loss(group_loss)

            # Gather all necessary data across devices
            tensors_to_gather = {
                'chosen': chosen_rewards,
                'rejected': rejected_rewards,
                'accuracies': reward_accuracies,
                'group_acc': group_acc.unsqueeze(0),
                'group_count': group_count.unsqueeze(0),
                'weights': weights.detach().unsqueeze(0),
                'margins': chosen_rewards - rejected_rewards
            }
            if loss_config.name in {'rdpo', 'ripo'}:
                tensors_to_gather['group_loss']=group_loss.detach().unsqueeze(0)
            gathered_tensors = {k: all_gather_if_needed(v, self.rank, self.world_size) for k, v in tensors_to_gather.items()}

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
            
        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps
            actual_loss=losses.mean()
            gathered_tensors={}
            
        elif loss_config.name == 'base':
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            logps_accuracies= (policy_chosen_logps > policy_rejected_logps).float()
            losses=-logps_accuracies
            group_idx=batch['group']
            group_base_acc, group_count = self.compute_group_sum_metric(logps_accuracies, group_idx)
            actual_loss=losses.mean()

            gathered_tensors = {
                'logps_accuracies': all_gather_if_needed(logps_accuracies, self.rank, self.world_size),
                'group_base_acc': all_gather_if_needed(group_base_acc.unsqueeze(0), self.rank, self.world_size),
                'group_count': all_gather_if_needed(group_count.unsqueeze(0), self.rank, self.world_size),
            }
            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
        if gathered_tensors:
            metrics.update({
                f'rewards_{train_test}/{k}': v.cpu().numpy().tolist() for k, v in gathered_tensors.items()
            })
        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return actual_loss, metrics

    
    def compute_group_metric(self, losses, group_idx, divide_by_totalcount=False):
        # compute observed counts and mean loss for each group
        group_idx= torch.LongTensor(group_idx).cuda()
        n_groups=torch.arange(self.n_groups)
        group_map = (group_idx == n_groups.unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        if divide_by_totalcount:
            group_loss = self.total_count*((group_map @ losses.view(-1))/self.group_counts)
        else:
            group_denom = group_count + (group_count==0).float() # avoid nans
            group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count
    
    def compute_group_sum_metric(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_idx= torch.LongTensor(group_idx).cuda()
        n_groups=torch.arange(self.n_groups)
        group_map = (group_idx == n_groups.unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_loss = (group_map @ losses.view(-1))
        return group_loss, group_count
    
    def compute_robust_loss(self, group_loss):
        adjusted_loss = group_loss.clone().detach()
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)#eqn 5 in paper--not needed for now
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        if self.config.loss.importance_sampling==False:

            self.adv_probs = self.adv_probs*torch.exp(self.config.loss.step_size*adjusted_loss)
            self.adv_probs = self.adv_probs/(self.adv_probs.sum()).float()
        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs
    def aggregate_worst_case_metrics(self,mean_eval_metrics:Dict[str,Union[float,List[float]]]):
        """Aggregate the worst case metrics from multiple datasets."""
        
        # Initialize a dictionary to store the worst case results
        worst_case_metrics = {}
        
        # Iterate over each metric in the first dataset as a base
        for metric_name in mean_eval_metrics[0].keys():
            # Extract the base name of the metric to group similar metrics (assuming metric_name ends with '_{integer}')
            base_metric_name = '_'.join(metric_name.split('_')[:-1])
            
            # Initialize a variable to store the worst case value
            worst_case_value = None
            
            # Determine if the metric is a loss or an accuracy type
            is_loss_metric = 'loss' in metric_name.lower()
            
            # Iterate over all datasets to find the worst case
            #print(mean_eval_metrics)
            for group_idx,eval_metrics in enumerate(mean_eval_metrics.values()):
                #print(f"{base_metric_name}_{group_idx}")
                metric_value = eval_metrics[f"{base_metric_name}_{group_idx}"]
                #print(metric_value)
                # Update the worst case value based on the metric type
                if worst_case_value is None:
                    worst_case_value = metric_value
                elif is_loss_metric and metric_value > worst_case_value:
                    worst_case_value = metric_value
                elif not is_loss_metric and metric_value < worst_case_value:
                    worst_case_value = metric_value
            #raise ValueError
            # Store the worst case result in the dictionary
            worst_case_metrics[f'worst_case_{base_metric_name}'] = worst_case_value
        
        return worst_case_metrics
    def aggregate_average_metrics(self, mean_eval_metrics: Dict[str, Union[float, List[float]]]):
        """Aggregate the average metrics from multiple datasets."""

        # Initialize a dictionary to store the average results
        average_metrics = {}

        # Iterate over each metric in the first dataset as a base
        for metric_name in mean_eval_metrics[0].keys():
            # Extract the base name of the metric to group similar metrics (assuming metric_name ends with '_{integer}')
            base_metric_name = '_'.join(metric_name.split('_')[:-1])

            # Initialize a variable to store the sum of metric values
            sum_metric_value = 0

            # Iterate over all datasets to calculate the sum of metric values
            for group_idx,eval_metrics in enumerate(mean_eval_metrics.values()):
                metric_value = eval_metrics[f"{base_metric_name}_{group_idx}"]
                sum_metric_value += metric_value
            length=group_idx+1
            # Calculate the average metric value
            average_metric_value = sum_metric_value / length

            # Store the average result in the dictionary
            average_metrics[f'average_{base_metric_name}'] = average_metric_value

        return average_metrics
    
    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""
        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
        patience=10*(192/self.config.eval_every)*self.config.patience_factor
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=patience, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=0, verbose=True)
        
        if self.config.loss.name in {'rdpo', 'ripo'} and getattr(self.config.loss, 'adaptive_step_size', False):
            self.initial_lr = min([param_group['lr'] for param_group in self.optimizer.param_groups])
            self.initial_step_size = self.config.loss.step_size  # Save initial step_size

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.is_jeopardy = self.config.datasets == ['jeopardy'] or 'jeopardy' in self.config.datasets[0]
        
        
        
        if self.config.loss.name in {'dpo', 'ipo', 'rdpo', 'ripo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        print(f"DTYPE: {next(self.policy.parameters()).dtype=}")


        if self.config.eval_only_once==True:
            cur_gpu_mem = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            print(f'currently allocated: {cur_gpu_mem / 1e9:.2f} GB')
            #### BEGIN EVALUATION ####
            rank0_print(f'Running evaluation after {self.example_counter} train examples')
            self.policy.eval()
            mean_eval_metrics={}
            for i in range(len(self.config.datasets)):
                mean_eval_metrics[i]=self.evaluate(eval_grp=f'test_{i}')
                    
            if self.config.eval_train_data==True:
                mean_train_metrics={}
                for i in range(len(self.config.datasets)):
                    mean_train_metrics[i]=self.evaluate(eval_grp=f'train_{i}')

            return
            

        for batch in self.train_iterator:
            cur_gpu_mem = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            print(f'currently allocated: {cur_gpu_mem / 1e9:.2f} GB')
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()
                mean_eval_metrics={}
                for i in range(len(self.config.datasets)):
                    mean_eval_metrics[i]=self.evaluate(eval_grp=f'test_{i}')

                worst_case_eval_metrics=self.aggregate_worst_case_metrics(mean_eval_metrics)
                self.log_worst_case_results(worst_case_eval_metrics, 'test')

                avg_case_eval_metrics=self.aggregate_average_metrics(mean_eval_metrics)
                self.log_average_results(avg_case_eval_metrics, 'test')
                    
                if self.example_counter > 0 and self.example_counter % self.config.save_every == 0 :
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir,[mean_eval_metrics])
                    
                #evalute training data
                if self.config.eval_train_data==True and self.example_counter % self.config.eval_train_every == 0:
                    mean_train_metrics={}
                    for i in range(len(self.config.datasets)):
                        mean_train_metrics[i]=self.evaluate(eval_grp=f'train_{i}')
                    worst_case_train_metrics=self.aggregate_worst_case_metrics(mean_train_metrics)
                    self.log_worst_case_results(worst_case_train_metrics, 'train')

                    avg_case_train_metrics=self.aggregate_average_metrics(mean_train_metrics)
                    self.log_average_results(avg_case_train_metrics, 'train')
                    #mean_eval_metrics_0=self.evaluate(eval_grp='train_0')
                    #mean_eval_metrics_1=self.batch_evaluate(eval_grp='train_1')
                
                if self.early_stopping:
                    mean_vald_metrics={}
                    for i in range(len(self.config.datasets)):
                        mean_vald_metrics[i]=self.evaluate(eval_grp=f'vald_{i}')
                    worst_case_vald_metrics=self.aggregate_worst_case_metrics(mean_vald_metrics)
                    self.log_worst_case_results(worst_case_vald_metrics, 'vald')

                    avg_case_vald_metrics=self.aggregate_average_metrics(mean_vald_metrics)
                    self.log_average_results(avg_case_vald_metrics, 'vald')

                    if self.config.scheduler_metric=='accuracy':
                        if self.config.loss.name in {'rdpo', 'ripo'}:
                            worst_case_vald_accuracies=worst_case_vald_metrics['worst_case_rewards_vald/accuracies']
                            self.scheduler.step(worst_case_vald_accuracies)
                        elif self.config.loss.name in {'dpo', 'ipo'}:
                            avg_case_vald_accuracies=avg_case_vald_metrics['average_rewards_vald/accuracies']
                            self.scheduler.step(avg_case_vald_accuracies)
                        else:
                            raise ValueError(f"Unknown loss function {self.config.loss.name}")
                    elif self.config.scheduler_metric=='loss':
                        if self.config.loss.name in {'rdpo', 'ripo'}:
                            worst_case_vald_losses=worst_case_vald_metrics['worst_case_loss/vald']
                            self.scheduler.step(-worst_case_vald_losses)
                        elif self.config.loss.name in {'dpo', 'ipo'}:
                            avg_case_vald_losses=avg_case_vald_metrics['average_loss/vald']
                            self.scheduler.step(-avg_case_vald_losses)
                            rank0_print('using average loss for scheduler')
                        else:
                            raise ValueError(f"Unknown loss function {self.config.loss.name}")
                    else:
                        raise ValueError(f"Unknown scheduler metric {self.config.scheduler_metric}")

                    # Check if any learning rate has fallen below the threshold
                    current_lr = min([group['lr'] for group in self.optimizer.param_groups])
                    rank0_print(current_lr, 'current learning rate')

                    if self.config.loss.name in {'rdpo', 'ripo'} and getattr(self.config.loss, 'adaptive_step_size', False):
                        lr_change_factor = current_lr / self.initial_lr
                        # Apply the same change factor to step_size if the learning rate has changed
                        #if lr_change_factor != 1:
                        new_step_size = self.initial_step_size * lr_change_factor
                        self.config.loss.step_size = new_step_size
                        print(f"Updated step_size to {new_step_size} due to LR change.")
                        # Update initial values to reflect current settings
                        self.initial_lr = current_lr
                        self.initial_step_size = new_step_size
                    if current_lr < self.config.min_lr*(1.001):
                        print(f"Stopping training as learning rate {current_lr} is below the threshold {self.config.min_lr}")
                        break
                    #mean_eval_metrics_0=self.evaluate(eval_grp='train_0')
                    #mean_eval_metrics_1=self.batch_evaluate(eval_grp='train_1')
            #### END EVALUATION ####
            
            #### POINT SELECTION ####            
            if self.data_selector is not None:
                                
                selected_batch, not_selected_batch, selected_size = self.data_selector.\
                    select_batch(batch, self.config.selected_batch_size,
                                 self.policy, self.reference_model)
                batch_size = selected_size
            
            else:
                selected_batch = batch
                not_selected_batch = None
                batch_size = self.config.batch_size


            torch.cuda.empty_cache()
            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            with torch.autograd.detect_anomaly():
                for microbatch_idx in range(self.config.gradient_accumulation_steps):
                    global_microbatch = slice_and_move_batch_for_device(selected_batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                    local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                    loss, metrics = self.get_group_batch_metrics(local_microbatch, self.config.loss, train=True)
                    if self.config.debug==True:
                        for param in self.policy.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                print("NaN gradient detected.")
                    (loss / self.config.gradient_accumulation_steps).backward()

                    for k, v in metrics.items():
                        batch_metrics[k].extend(v)
                    del global_microbatch
                    del local_microbatch

                grad_norm = self.clip_gradient()
                self.optimizer.step()
                #self.scheduler.step()
                self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                #mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics = {}
                if self.config.loss.name not in {'sft'}:
                    gc=batch_metrics['rewards_train/group_count']
                    gc_n=np.array(gc)
                    gc_n=gc_n.sum(axis=0)
                #print(batch_metrics)
                for k,v in batch_metrics.items():
                    if k in {'rewards_train/group_acc','rewards_train/group_loss'}:
                        v_n=np.array(v)
                        #print(v,v_n,v_n.shape)
                        v_n=v_n.sum(axis=0)
                        v_n=np.divide(v_n,gc_n)
                        mean_train_metrics[k]= v_n.tolist()
                        for i in range(len(v_n)):
                            mean_train_metrics[f'{k}_{i}']= v_n[i]
                    elif k == 'rewards_train/group_count':
                        v_n=np.array(v)
                        #print(v,v_n,v_n.shape)
                        v_n=v_n.sum(axis=0)
                        mean_train_metrics[k]= v_n.tolist()
                        for i in range(len(v_n)):
                            mean_train_metrics[f'{k}_{i}']= v_n[i]
                    elif k=='rewards_train/weights':
                        v_n=np.array(v)
                        #print(v,v_n,v_n.shape)
                        v_n=v_n.mean(axis=0)
                        mean_train_metrics[k]= v_n.tolist()
                        for i in range(len(v_n)):
                            mean_train_metrics[f'{k}_{i}']= v_n[i]
                    else:
                        mean_train_metrics[k]= sum(v) / len(v)
                              
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            max_gpu_mem_so_far = torch.cuda.max_memory_allocated()
            print(f"Max allocated so far: {max_gpu_mem_so_far / 1e9:.2f} GB")
            cur_gpu_mem = torch.cuda.memory_allocated()
            print(f'currently allocated (after train): {cur_gpu_mem / 1e9:.2f} GB')
            torch.cuda.reset_peak_memory_stats()
            if self.config.max_train_examples is not None and self.example_counter > self.config.max_train_examples:
                break
            #### END TRAINING ####




        # evaluate one last time after training
        #self.evaluate()
        self.policy.eval()
        mean_eval_metrics={}
        for i in range(len(self.config.datasets)):
            mean_eval_metrics[i]=self.evaluate(eval_grp=f'test_{i}')

         
        worst_case_eval_metrics=self.aggregate_worst_case_metrics(mean_eval_metrics)
        self.log_worst_case_results(worst_case_eval_metrics, 'test')

        avg_case_eval_metrics=self.aggregate_average_metrics(mean_eval_metrics)
        self.log_average_results(avg_case_eval_metrics, 'test')

        if self.config.eval_train_end==True:
            mean_train_metrics={}
            for i in range(len(self.config.datasets)):
                mean_train_metrics[i]=self.evaluate(eval_grp=f'train_{i}')
            
            worst_case_train_metrics=self.aggregate_worst_case_metrics(mean_train_metrics)
            self.log_worst_case_results(worst_case_train_metrics, 'train')

            avg_case_train_metrics=self.aggregate_average_metrics(mean_train_metrics)
            self.log_average_results(avg_case_train_metrics, 'train')
   
    def evaluate(self,eval_grp:str):
        train_test, group_id = eval_grp.split("_")
        current_batch, train = self.get_current_batch(train_test, int(group_id))
        self.log_gpu_memory("currently allocated")
        mean_eval_metrics_1 = self.compute_metrics(current_batch, group_id, train)
        if train_test == 'vald':
            mean_eval_metrics_1 = {k.replace('eval', 'vald'): v for k, v in mean_eval_metrics_1.items()}
        rank0_print(f'{eval_grp} after {self.example_counter}: {formatted_dict(mean_eval_metrics_1)}')
        
        current_sample_batch, train = self.get_current_sample_batch(train_test, int(group_id))
        self.sample_during_eval(current_sample_batch, group_id)
        self.log_gpu_memory("Max allocated so far", peak=True)
        self.log_results(mean_eval_metrics_1, eval_grp, group_id)
                    
        rank0_print(f'finish dataset-{eval_grp}')
        return mean_eval_metrics_1
    
    def get_current_batch(self, train_test, group_id):
        if train_test == 'test':
            return self.eval_batches[group_id], False
        elif train_test == 'train':
            return self.traineval_batches[group_id], True
        elif train_test == 'vald':
            return self.vald_batches[group_id], False
        else:
            raise NotImplementedError
    def get_current_sample_batch(self, train_test, group_id):
        if train_test == 'test':
            return self.gen_batches[group_id], False
        elif train_test == 'train':
            return self.traingen_batches[group_id], True
        elif train_test == 'vald':
            return self.valdgen_batches[group_id], False
        else:
            raise NotImplementedError
        
    def log_gpu_memory(self, message, peak=False):
        mem_usage = torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()
        print(f"{message}: {mem_usage / 1e9:.2f}GB")
        torch.cuda.empty_cache()
        if not peak:
            torch.cuda.reset_peak_memory_stats()

    def compute_metrics(self, current_batch, group_id, train):
        all_eval_metrics = defaultdict(list) if self.config.use_ref else defaultdict(list)
        
        if self.config.use_ref:
            for eval_batch in tqdm.tqdm(current_batch, desc='Computing eval metrics', disable=self.rank != 0):
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                with torch.no_grad():
                    _, eval_metrics = self.get_batch_metrics(local_eval_batch,self.config.loss, train=train)
                for k, v in eval_metrics.items():
                    all_eval_metrics[k].extend(v)

                del local_eval_batch
                del eval_batch

            mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
            mean_eval_metrics_1 = {f'{k}_{group_id}': val for k, val in mean_eval_metrics.items()}
        else:
            mean_eval_metrics_1 = {'use_ref': self.config.use_ref}

        return mean_eval_metrics_1
    
    def sample_during_eval(self, current_batch, group_id):
        if not self.config.sample_during_eval:
            return
        
        sample_batches = self.get_sample_batches(current_batch)
        for eval_batch in tqdm.tqdm(sample_batches, desc='Generating samples...', disable=self.rank != 0):
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)
            self.log_samples(policy_samples, reference_samples, eval_batch, group_id)
        del eval_batch
        del local_eval_batch

    def get_sample_batches(self, current_batch):
        if self.config.n_eval_model_samples < self.config.eval_batch_size:
                rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                sample_batches = current_batch[:1]
        else:
            n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
            sample_batches = current_batch[:n_sample_batches]
        return sample_batches
    
    def log_samples(self, policy_samples, reference_samples, eval_batch, group_id):
        sample_dir = os.path.join(self.run_dir, f'step-{self.example_counter}_samples')
        os.makedirs(sample_dir, exist_ok=True)
        policy_samples_path = os.path.join(sample_dir, f"policy_samples_{group_id}.csv")
        reference_samples_path = os.path.join(sample_dir, f"reference_samples_{group_id}.csv")

        # Save policy samples
        with open(policy_samples_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["step", "prompt", "sample","correct response","group id"])
            for prompt, sample, correct_response in zip(eval_batch['prompt'], policy_samples, eval_batch['chosen_response_only']):
                writer.writerow([self.example_counter,prompt, sample, correct_response,group_id])

        # Log to wandb if enabled
        if self.config.wandb.enabled and self.rank == 0:
            policy_table = wandb.Table(columns=["step", "prompt", "sample","correct response","group id"])
            for prompt, sample, correct_response in zip(eval_batch['prompt'], policy_samples, eval_batch['chosen_response_only']):
                policy_table.add_data(self.example_counter,prompt, sample, correct_response,group_id)
            wandb.log({f"policy_samples_group_{group_id}": policy_table}, step=self.example_counter)



        # Save reference samples if applicable
        if self.config.loss.name == 'rdpo' and self.config.ref_sample:
            with open(reference_samples_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["step", "prompt", "sample","correct response","group id"])
                for prompt, sample, correct_response in zip(eval_batch['prompt'], reference_samples, eval_batch['chosen_response_only']):
                    writer.writerow([prompt, sample, correct_response])

            if self.config.wandb.enabled and self.rank == 0:
                reference_table = wandb.Table(columns=["step", "prompt", "sample","correct response","group id"])
                for prompt, sample, correct_response in zip(eval_batch['prompt'], reference_samples, eval_batch['chosen_response_only']):
                    reference_table.add_data(self.example_counter,prompt, sample, correct_response,group_id)
                wandb.log({f"reference_samples_group_{group_id}": reference_table}, step=self.example_counter)
    
    def log_results(self, mean_eval_metrics, eval_grp, group_id):
        """Logs evaluation results to different sinks such as Weights & Biases and local CSV files."""
        # Log to Weights & Biases if enabled and if the current process is rank 0 (to avoid duplicate logs in multi-GPU setups)
        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(mean_eval_metrics, step=self.example_counter)
        
        # Define CSV file path
        results_csv_path = os.path.join(self.run_dir, f"{self.config.datasets[0:1]}_experiment_results.csv")
        
        # Check if the file exists to decide whether to write headers
        file_exists = os.path.exists(results_csv_path)
        
        # Write results to the CSV file
        with open(results_csv_path, mode='a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:  # Write headers if the file does not exist
                headers = ["Experiment Name", "Group ID"] + list(mean_eval_metrics.keys())
                writer.writerow(headers)
            
            # Prepare the row to be written
            row = [self.config.exp_name, group_id] + list(mean_eval_metrics.values())
            writer.writerow(row)
    
        print(f"Logged results for {eval_grp} with metrics: {mean_eval_metrics}")



    def log_worst_case_results(self, worst_case_metrics, eval_grp):
        """Logs worst-case evaluation results to different sinks such as Weights & Biases and local CSV files."""
        # Log to Weights & Biases if enabled and if the current process is rank 0 (to avoid duplicate logs in multi-GPU setups)
        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(worst_case_metrics, step=self.example_counter)

        # Define CSV file path for worst-case metrics
        results_csv_path = os.path.join(self.run_dir, f"{self.config.datasets[0]}_worst_case_results.csv")

        # Check if the file exists to decide whether to write headers
        file_exists = os.path.exists(results_csv_path)

        # Write results to the CSV file
        with open(results_csv_path, mode='a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:  # Write headers if the file does not exist
                headers = ["Experiment Name"] + list(worst_case_metrics.keys())
                writer.writerow(headers)

            # Prepare the row to be written
            row = [self.config.exp_name] + list(worst_case_metrics.values())
            writer.writerow(row)

        print(f"Logged worst-case results for {eval_grp} with metrics: {worst_case_metrics}")

    def log_average_results(self, average_metrics, eval_grp):
        """Logs average evaluation results to different sinks such as Weights & Biases and local CSV files."""
        # Log to Weights & Biases if enabled and if the current process is rank 0 (to avoid duplicate logs in multi-GPU setups)
        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(average_metrics, step=self.example_counter)

        # Define CSV file path for average metrics
        results_csv_path = os.path.join(self.run_dir, f"{self.config.datasets[0]}_average_results.csv")

        # Check if the file exists to decide whether to write headers
        file_exists = os.path.exists(results_csv_path)

        # Write results to the CSV file
        with open(results_csv_path, mode='a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:  # Write headers if the file does not exist
                headers = ["Experiment Name"] + list(average_metrics.keys())
                writer.writerow(headers)

            # Prepare the row to be written
            row = [self.config.exp_name] + list(average_metrics.values())
            writer.writerow(row)

        print(f"Logged average results for {eval_grp} with metrics: {average_metrics}")

