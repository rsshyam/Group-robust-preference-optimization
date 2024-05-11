import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from src.utils import get_local_dir, get_local_run_dir, get_local_run_dir_group, disable_dropout, init_distributed, get_open_port
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


OmegaConf.register_new_resolver("get_local_run_dir_group", lambda exp_name, group_name, local_dirs: get_local_run_dir_group(exp_name, group_name, local_dirs))


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
        tags=[f"n_epochs_{config.n_epochs}",f"learning_rate_{config.lr}",f"batch_size_{config.batch_size}"]
        if 'po' in config.loss.name:
            tags.append(f"beta_{config.loss.beta}") 
        wandb.init(
            group=config.group_name,
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
            tags=tags
        )
        

    #TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer=get_trainer(config.trainer,policy, config, config.seed, config.local_run_dir, reference_model=reference_model,data_selector=data_selector, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    if config.loss.name in {'dpo','ipo'}:
        exp_name=f"{config.loss.name}_beta_{config.loss.beta}_seed_{config.seed}_batch_{config.batch_size}_nepoch_{config.n_epochs}_lr_{config.lr}_avg_fr_vald"
    elif config.loss.name in {'sft','base'}:
        exp_name=f"{config.loss.name}_seed_{config.seed}_batch_{config.batch_size}_nepoch_{config.n_epochs}_lr_{config.lr}"
    elif config.loss.name in {'rdpo','ripo'}:
        exp_name=f"{config.loss.name}_beta_{config.loss.beta}_seed_{config.seed}_expstepsize_{config.loss.step_size}_nepoch_{config.n_epochs}_batch_{config.batch_size}_weightedbatch_{config.weighted_batches}"
    else:
        raise NotImplementedError
    config.exp_name=exp_name
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    dataset_group = config.datasets[0].split('_')[0]
    #print(f'{dataset_group}_{len(config.datasets)}'+f'tr_frac{config.train_frac}'+f'{config.model.name}_spairs_{config.sep_pairs}_{config.trainer}')
    #if config.new_grp_name:
    group_indices="_".join(dataset.split('_')[1] for dataset in config.datasets)
    group=f'{dataset_group}_{group_indices}'+f'tr_frac{config.train_frac}'+f'{config.model.name_or_path}_spairs_{config.sep_pairs}_{config.trainer}'
    config.group_name=group
    print(group)
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