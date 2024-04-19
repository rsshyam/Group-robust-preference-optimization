import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from src.utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
from src.trainers_factory import get_trainer
import wandb
import json
import socket
from typing import Optional, Set
from src.models import ModelGenerator
from src.data_selection import DataSelector


#System specific installs:
if os.name != 'nt':
    #We can't install resource on windows
    import resource


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module,
                reference_model: Optional[nn.Module] = None, data_selector: Optional[DataSelector] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.login(key=config.wandb.key)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )
        '''
        tags=[f"num_epochs_{config.num_epochs}",f"learning_rate_{congfig.lr}",f"beta_{config.loss.beta}"]
        if config.loss in {'dpo','ipo'}:
            exp_name=config.model+config.datasets+.wandb_name +"_"+ + "_" + str(args.seed)
        elif config.loss='sft':
            exp_name=
        
        else:
            exp_name=args.wandb_name +"_"+args.dpo_type + "_" + str(args.rdpo_exp_step_size) +"_" + str(args.rdpo_batch_size) + '_' + str(args.rdpo_weighted_batches) + "_" + args.rdpo_adj  + "_" + str(args.seed)
        wandb.init(
            group=f'state_dim{args.state_dim}'+f'action_num{args.action_num}'+f'group_num{args.group_num}'+f'pref_data_num{args.pref_data_num}'+f'weights{args.weights}'+f'feature_type{args.feature_type}'+f'eval_metric{args.eval_metric}_state-1',
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=args.__dict__,
            dir=log_dir,
            name=exp_name,
            tags=tags
        '''

    #TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer=get_trainer(config.trainer,policy, config, config.seed, config.local_run_dir, reference_model=reference_model,data_selector=data_selector, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    #CREATES THE POLICY
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)

    print('build data selector')     
    data_selector = hydra.utils.instantiate(config.get('data_selection', None),
                                            other_config=config,
                                            _recursive_=False)

    print('building policy')
    #TODO: Temporary code -> we can store all models in a class and request access to them as needed in the trainer
    model_generator = ModelGenerator()
    models = model_generator.generate_models(config)  
    
    if config.loss.name == 'sft':
        policy = models.get('sft_model', None)
        reference_model = None
    elif config.loss.name == 'base':
        policy = models.get('base_model', None)
        reference_model = None
    else:
        policy = models.get('policy_model', None)
        reference_model = models.get('ref_model', None)  
            
    if 'FSDP' in config.trainer and os.name != 'nt':
        
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model, data_selector), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model, data_selector=data_selector)


if __name__ == '__main__':
    main()