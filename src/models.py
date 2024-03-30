# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:04:33 2024

@author: William
"""

import json

import torch
import transformers
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

from src.utils import get_local_dir
from omegaconf.listconfig import ListConfig

class ModelGenerator:
    
    
    def load_saved_model(self, model, model_state_dict_path):
        
        #Load the state dictionary into memory
        state_dict = torch.load(model_state_dict_path, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']

        #Load state dict into policy and ref model:
        print(f'loading pre-trained weights at step {step} from\
              {model_state_dict_path} with metrics {json.dumps(metrics, indent=2)}')
        
        #load_state_dict moves weights onto the model's device
        model.load_state_dict(state_dict['state'])
        
        return model 
    
    def create_policy_from_config(self, model_config, trainer:str, local_dirs, reference:bool=False):
        
        model_kwargs = {'device_map': 'balanced'} if trainer == 'BasicTrainer' else {}
        
        if reference:
            dtype = model_config.policy_dtype
        else:
            dtype = model_config.reference_dtype
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype
        )
        
        #Load model from Huggingface:
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            model_config.name_or_path, 
            cache_dir=local_dirs, 
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            output_hidden_states=True,
            trust_remote_code=True,
            **model_kwargs)
        
        policy.gradient_checkpointing_enable()
        
        #Setup model with LoRA:
        if model_config.use_lora: 
            policy = prepare_model_for_kbit_training(policy)
            
            target_modules = model_config.lora_target_modules
            
            assert isinstance(target_modules, ListConfig) or isinstance(target_modules, list),\
                f'lora_target_modules type:{type(target_modules)} must be type ListConfig or list'
    
            loraconfig = LoraConfig(
                r=model_config.lora_rank,
                lora_alpha=model_config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=model_config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM")
    
            #Apply lora config to policy model:
            policy = get_peft_model(policy, loraconfig)            
            policy = self.manually_map_lora_to_dtype(policy, getattr(torch,dtype))
                    
        print('Current GPU usage')
        
        for dev in range(torch.cuda.device_count()):
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(dev)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(dev)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(dev)/1024/1024/1024))
            
        print(f'Loaded model onto device: {policy.device}')
        
        return policy
        
        
                
    def create_policy(self, model_name, dtype, config, use_lora:bool=False,
                      lora_rank:int=8, lora_alpha:int=32, lora_dropout:float=0.0):
        """
        Load a model from huggingface AutoModelForCausalLLM, apply a bitsandbytes
        config file and if required setup the model to use lora.

        Parameters
        ----------
        model_name : str
            Huggingface model name or path
        dtype : <torch.dtype>
            float point precision to map the weights of the loaded model to.
        use_lora : bool, optional
            DESCRIPTION. The default is False.
        lora_rank : int, optional
            DESCRIPTION. The default is 8.
        lora_alpha : int, optional
            DESCRIPTION. The default is 32.
        lora_dropout : float, optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        policy : nn.Module
            Huggingface LLM module setup with dtype precision and lora training weights

        """
        #Setup model and bitsandbytes conifgs:
        model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
        
        compute_dtype = getattr(torch, dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype
        )
        
        #Load model from Huggingface:
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=get_local_dir(config.local_dirs), 
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            output_hidden_states=True,
            trust_remote_code=True,
            **model_kwargs)
        
        policy.gradient_checkpointing_enable()
        
        #Setup model with LoRA:
        if use_lora: 
            policy = prepare_model_for_kbit_training(policy)
            
            target_modules = config.model.lora_target_modules
            
            assert isinstance(target_modules, ListConfig) or isinstance(target_modules, list),\
                f'lora_target_modules type:{type(target_modules)} must be type ListConfig or list'
    
            loraconfig = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM")
    
            #Apply lora config to policy model:
            policy = get_peft_model(policy, loraconfig)
            policy = self.manually_map_lora_to_dtype(policy, getattr(torch,dtype))
        
            
        print('Current GPU usage')
        
        for dev in range(torch.cuda.device_count()):
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(dev)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(dev)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(dev)/1024/1024/1024))
            
        print(f'Loaded model onto device: {policy.device}')
        
        return policy
        
    def manually_map_lora_to_dtype(self, policy, dtype):
        """
        Maps a model setup with LoRA layers to the specified dtype. This is used
        after the lora config has been applied to a huggingface model loaded with a 
        specific dtype. 
        
        Parameters
        ----------
        policy : TYPE
            DESCRIPTION.
        dtype : TYPE
            DESCRIPTION.

        Returns
        -------
        policy : TYPE
            DESCRIPTION.

        """
                
        for name, module in policy.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(dtype)
            if 'norm' in name:
                module = module.to(dtype)
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(dtype)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if module.weight.dtype == torch.float32:
                        module = module.to(dtype)
        
        return policy
           
    
    def generate_models(self, config):
        """
        Return a dictionary with the relevant models create and sorted

        Parameters
        ----------
        config : dict
            Config file containing the relevant parameters

        Raises
        ------
        NotImplementedError
            The config.loss.name has not been implemented yet

        Returns
        -------
        models : dict
            A dictionary of policy models
        """
              
        if config.loss.name == 'sft':
            
            sft_model = self.create_policy(config.model.name_or_path, 
                                          config.model.policy_dtype,
                                          config,
                                          use_lora=config.model.use_lora,
                                          lora_rank=config.model.lora_rank,
                                          lora_alpha=config.model.lora_alpha,
                                          lora_dropout=config.model.lora_dropout)
            
            models = {'sft_model': sft_model}
        
        elif config.loss.name in ['dpo', 'ipo']:
            
            #create main policy:
            policy_model = self.create_policy(config.model.name_or_path, 
                                              config.model.policy_dtype,
                                              config,
                                              use_lora=config.model.use_lora,
                                              lora_rank=config.model.lora_rank,
                                              lora_alpha=config.model.lora_alpha,
                                              lora_dropout=config.model.lora_dropout)
            
            #create the reference policy:
            ref_model = self.create_policy(config.model.name_or_path, 
                                              config.model.reference_dtype,
                                              config,
                                              use_lora=config.model.use_lora,
                                              lora_rank=config.model.lora_rank,
                                              lora_alpha=config.model.lora_alpha,
                                              lora_dropout=config.model.lora_dropout)
            
            policy_device = policy_model.device
            ref_device = ref_model.device
            
            #Check sft model is asserted:
            if config.assert_sft_step:
                assert config.model.archive is not None,\
                    'config.model.archive should be provided when training with PO methods'
            
            #Load the previous model state dict and upload to policy and reference model:
            if config.model.archive is not None:
                
                #Load state dict:
                state_dict = torch.load(config.model.archive, map_location='cpu')
                step, metrics = state_dict['step_idx'], state_dict['metrics']
        
                #Load state dict into policy and ref model:
                print(f'loading pre-trained weights at step {step} from\
                      {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
                policy_model.load_state_dict(state_dict['state'])
                ref_model.load_state_dict(state_dict['state'])
                
                print('Loaded pretrained weights')
            
            #Ensure the device hasn't changed at this step:
            assert (policy_model.device == policy_device) and (ref_model.device == ref_device), \
                'The policy and reference models device should not change'
            
            models = {'policy_model': policy_model,
                      'ref_model': ref_model}
            
        else:
            raise NotImplementedError(
                f'config.loss.name: {config.loss.name} not implemented yet')
        
        return models

if __name__ == '__main__':
    
    #TODO: Write some tests:
    print('Model Generator Tests')
    
    