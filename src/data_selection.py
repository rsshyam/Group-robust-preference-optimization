# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:46:17 2024

@author: William
"""

import torch

from abc import ABC, abstractmethod
from collections.abc import Iterable

from src.utils import rank0_print
       
class DataSelector(ABC):
    
    """
    Abstract base class for the different data selection functions in our code 
    base, we can add and adapt this as necessary but it is probably overkill.
    """
      
    @abstractmethod
    def select_batch(self, batch:Iterable, selected_batch_size:int) -> Iterable:
        pass
    
class UniformRandomSelection(DataSelector):
    
    """
    Randomly select and return a subset of the input batch.
    
    """
    
    def subselect_batch(self, batch:dict, selected_idx:torch.tensor, 
                        not_selected_idx:torch.tensor):
        """
        Select a subset of the batch, return the selected and not selected subsets.

        """
        
        selected_batch = dict()
        not_selected_batch = dict()
                
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
        
    def batch_len(self, batch):
        """
        Return the length of a list of the first key.
        
        """
        
        keys = list(batch.keys())
        
        return len(batch[keys[0]])
    
    def select_batch(self, batch:Iterable, selected_batch_size:int) -> Iterable:
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
        
            
        
        