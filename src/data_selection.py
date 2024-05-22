# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from collections.abc import Iterable

from src.utils import ( 
    get_local_dir,
    slice_and_move_batch_for_device)
from src.models import ModelGenerator
from src.loss_utils import (
    preference_loss,
    concatenated_forward)

from omegaconf import DictConfig
from typing import Dict, List, Union
       
class DataSelector(ABC):
    
    """
    Abstract base class for the different data selection functions in our code 
    base, we can add and adapt this as necessary but it is probably overkill.
    """
    
    def __init__(self, other_config:str):
        self.config = other_config
          
    def select_top_k(self, vector, k):
    
        sorted_idx = torch.argsort(vector, descending=True)

        top_x_indices = sorted_idx[:k]
        other_indices = sorted_idx[k:]

        return top_x_indices, other_indices
    
    def subselect_batch(self, batch:dict, selected_idx:torch.tensor, 
                        not_selected_idx:torch.tensor):
        """
        Select a subset of the batch, return the selected and not selected subsets.

        """
        
        selected_batch = dict()
        not_selected_batch = dict()
          
        #We can try use this:
        #sliced = {k: v[start:end] for k, v in batch.items()} only works for consecutive elements
        
        for key in batch.keys():
            
            key_batch = batch[key]
            selected_batch[key] = [key_batch[i] for i in selected_idx.to(dtype=torch.long)]
            
            #If the batch stores as type tensor then map to tensor:
            if isinstance(key_batch, torch.Tensor):
                selected_batch[key] = torch.stack(selected_batch[key])
            
            
            if not_selected_idx is not None:
                not_selected_batch[key] = [key_batch[i] for i in not_selected_idx.to(dtype=torch.long)]
                
                #If the data is stored as a tensor then map to tensor:
                if isinstance(key_batch, torch.Tensor):
                    not_selected_batch[key] = torch.stack(not_selected_batch[key])
                
            else:
                not_selected_batch = None
                
        return selected_batch, not_selected_batch
        
    @abstractmethod
    def select_batch(self, batch:dict, selected_batch_size:int, **kwargs) -> Iterable:
        pass
    
class UniformRandomSelection(DataSelector):
    
    """
    Randomly select and return a subset of the input batch.
    
    """
    
    def __init__(self, other_config):
        pass
        
    def batch_len(self, batch):
        """
        Return the length of a list of the first key.
        
        """
        
        keys = list(batch.keys())
        
        return len(batch[keys[0]])
    
    def select_batch(self, batch:Iterable, selected_batch_size:int,
                     policy:nn.Module=None, ref_policy:nn.Module=None) -> Iterable:
        """
        Return the random/uniform selected batch and not selected batch.

        """
        
        blen = self.batch_len(batch)
        
        if selected_batch_size > blen:
            print('selected batch size:{selected_batch_size} is greater than batch size:{blen}')
            selected_batch_size = blen
        
        idx = torch.randperm(blen)
        
        selected, not_selected = self.subselect_batch(batch, idx[:selected_batch_size],
                                      None if selected_batch_size == blen \
                                      else idx[selected_batch_size:])
              
        return selected, not_selected, selected_batch_size
        
      
class SFTRHOLossSelection(DataSelector):
    pass
      
class DPORHOLossSelection(DataSelector):
    
    """
    Selects and returns a subset of the input batch using the RHO-Loss selection
    objective:
        
        RHO(x,y) = L(x,y) - L_ref(x,y)
        
    There are two options:
        1. Memory efficient but Compute slow
        
        We use two forward passes, one to calculate the rho objective without
        creating a computation graph and one with .train() which does create a
        computation graph but only on the small selected batch.
        
        2. Memory Using but Compute fast.
        
        We use a single forward pass that creates a computation graph for the
        entire batch, we then uses the losses calculated from this to select a
        sub-batch and then the backprop is only applied to those elements of the
        graph - does multiplying by zero at the loss stage prevent gradient being
        calculated any further?
        
        We also want to be able to do gradient accumulation steps
        
    Aim to implement both? -> might need to adjust depending upon FSDP
    How do we test FSDP given locally we only have 1 GPU?
    """
    
    def __init__(self, ft_state_dict_path, sft_state_dict_path, model, other_config):
        
        """
        Using the config, create the sft and ft reference models
        """
       
        #For FSDP we'll need to take in a model and then shard it on each process
        #see FSDP trainer script 
        super().__init__(other_config)
        
        #local_dir = get_local_dir(other_config.local_dirs)
        local_dir = get_local_dir(self.config.local_dirs)
        trainer = self.config.get('trainer', 'BasicTrainer')
        
        model_generator = ModelGenerator()
        sft_ref_model = model_generator.\
            create_policy_from_config(model, trainer=trainer, 
                                      local_dirs=local_dir,
                                      reference=True)
        
        
        ft_ref_model = model_generator.\
            create_policy_from_config(model, trainer=trainer,
                                      local_dirs=local_dir,
                                      reference=True)
        
        self.sft_ref_model = model_generator.load_saved_model( 
                sft_ref_model, sft_state_dict_path)
        
        self.ft_ref_model = model_generator.load_saved_model(
                ft_ref_model, ft_state_dict_path)
        
        
    def get_batch_preference_loss(self, ft_model: nn.Module, sft_model: nn.Module, 
                                  batch: Dict[str, Union[List, torch.LongTensor]],
                                  loss_config: DictConfig):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""
        
        with torch.no_grad():
                
            #Implement gradient accumulation without FSDP compatability at this point:
            accumulated_losses = list()
            
            for i in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, i,
                                                                    self.config.gradient_accumulation_steps, 0)
    
                #TODO: Implement FSDP calculation here for RHO-Loss policies etc...
            
                policy_chosen_logps, policy_rejected_logps = concatenated_forward(ft_model, global_microbatch)
                reference_chosen_logps, reference_rejected_logps = concatenated_forward(sft_model, global_microbatch)
    
                if loss_config.name == 'dpo':
                    loss_kwargs = {'beta': loss_config.beta,
                                   'reference_free': loss_config.reference_free,
                                   'label_smoothing': loss_config.label_smoothing,
                                   'ipo': False}
                elif loss_config.name == 'ipo':
                    loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
                else:
                    raise ValueError(f'unknown loss {loss_config.name}')
    
                losses, _, _ = preference_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    reference_chosen_logps, reference_rejected_logps,
                    **loss_kwargs)
                
                accumulated_losses.append(losses)
            
        #Return the losses accumulated over grad accum steps
        return torch.concat(accumulated_losses).to(device=losses.device)
        
    def select_batch(self, batch:dict, selected_batch_size:int,
                     policy:nn.Module, ref_policy:nn.Module) -> Iterable:
               
        #Calculate the batch length and adjust selected batch size:
        blen = len(list(batch.values())[0])
        if selected_batch_size > blen:
            print('selected batch size:{selected_batch_size} is greater than batch size:{blen}')
            selected_batch_size = blen
        
        #Calculate the ref model losses for the batch:
        ref_model_loss = self.get_batch_preference_loss(ft_model=self.ft_ref_model,
                                                        sft_model=self.sft_ref_model,
                                                        batch=batch,
                                                        loss_config=self.config.loss)
    
        
        #These look wrong in initial implementation
        model_loss = self.get_batch_preference_loss(ft_model=policy,
                                                    sft_model=ref_policy,
                                                    batch=batch,
                                                    loss_config=self.config.loss)
                
        rho_loss = model_loss - ref_model_loss
        selected_idx, not_selected_idx = self.select_top_k(rho_loss, selected_batch_size)
        
        selected, not_selected = self.subselect_batch(batch, selected_idx,
                                      None if selected_batch_size == blen \
                                      else not_selected_idx)
        
        return selected, not_selected, selected_batch_size
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    