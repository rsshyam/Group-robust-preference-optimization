sh scripts/run_multi_robust.sh --model gemma-7b --eval_every 960 --use_kfoldsplit True --adaptive_step_size True --datasets goqa_0,goqa_1,goqa_2,goqa_3,goqa_4 --model_archive /scratch/uceesr4/.cache/uceesr4/goqa_0_1_2_3_4tr_frac0.8google/gemma-7b_spairs_False_GroupTrainer/sft_seed_0_batch_16_nepoch_1_lr_0.0001_2024-05-08_17-42-05_815286/LATEST/policy.pt --trainer GroupTrainerEarlyStop
