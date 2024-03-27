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
    
    def subselect_batch(self, batch, selected_idx, not_selected_idx):
        
        selected_batch = dict()
        not_selected_batch = dict()
                
        for key in batch.keys():
            
            selected_batch = [batch[key][i] for i in selected_idx.to(dtype=torch.long)]
            
            if not_selected_idx is not None:
                not_selected_batch = [batch[key][i] for i in not_selected_idx.to(dtype=torch.long)]
            else:
                not_selected_batch = None
                
        return selected_batch, not_selected_batch
        
    def batch_len(self, batch):
        
        keys = list(batch.keys())
        
        return len(batch[keys[0]])
    
    def select_batch(self, batch:Iterable, selected_batch_size:int) -> Iterable:
        
        blen = self.batch_len(batch)
        
        if selected_batch_size > blen:
            print('selected batch size:{selected_batch_size} is greater than batch size:{blen}')
            selected_batch_size = blen
        
        idx = torch.randperm(blen)
        
        return self.subselect_batch(batch, idx[:selected_batch_size],
                                    None if selected_batch_size == blen \
                                        else idx[selected_batch_size:])        
              
        
            
        
        