# trainer_factory.py
from src.trainers.basictrainer import BasicTrainer
from src.trainers.grouptrainer import GroupTrainer
from src.trainers.paralleltrainer import FSDPTrainer,TensorParallelTrainer
from src.trainers.grouptrainerdebug import GroupTrainerDebug

def get_trainer(trainer,policy, config, seed, local_run_dir, reference_model, data_selector, rank, world_size):
    if trainer == "BasicTrainer":
        return BasicTrainer(policy, config, seed, local_run_dir, reference_model,data_selector, rank, world_size)
    elif trainer == "GroupTrainer":
        return GroupTrainer(policy, config, seed, local_run_dir, reference_model,data_selector, rank, world_size)
    elif trainer == "GroupTrainerDebug":
        return GroupTrainerDebug(policy, config, seed, local_run_dir, reference_model,data_selector, rank, world_size)
    elif trainer == "parallel_fsdp":
        return FSDPTrainer(policy, config, seed, local_run_dir, reference_model,data_selector, rank, world_size)
    elif trainer == "parallel_tensor":
        return TensorParallelTrainer(policy, config, seed, local_run_dir, reference_model,data_selector, rank, world_size)
    else:
        raise ValueError("Unknown trainer type")