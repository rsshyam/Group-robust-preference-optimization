#!/bin/bash

# Default parameters
MODEL="phi-3-mini4kinstr"
DATASETS="oqa_SEX_Male,oqa_SEX_Female"
TRAIN_FRAC=0.8
LOSS="dpo" # ipo, ripo, rdpo
GRADIENT_ACCUMULATION_STEPS=2
BATCH_SIZE=16
EVAL_BATCH_SIZE=8
SAMPLE_DURING_EVAL="False"
TRAINER="GroupTrainer"
LR=1e-4

MODEL_ARCHIVE="/scratch/zceeich/.cache/zceeich/sft_seed_0_batch_16_nepoch_1_lr_0.0001_2024-04-30_00-20-49_171715/LATEST/policy.pt"    ## SFT Sex2Group 1pair best-rand
#MODEL_ARCHIVE="/scratch/zceeich/.cache/zceeich/sft_seed_0_batch_16_nepoch_1_lr_0.0001_2024-04-29_18-09-35_090713/LATEST/policy.pt"   ## SFT Sex2Group 1pair best-worst

#"/scratch/uceesr4/.cache/uceesr4/sft_seed_0_batch_16_nepoch_1_lr_0.0001_2024-04-21_20-47-58_027081/LATEST/policy.pt" 0,1
# /scratch/uceesr4/.cache/uceesr4/sft_seed_0_batch_16_nepoch_1_lr_0.0001_2024-04-23_18-32-57_130315/LATEST/policy.pt 0,5
# /scratch/uceesr4/.cache/uceesr4/sft_seed_0_batch_16_nepoch_1_lr_0.0001_2024-04-25_15-52-35_455524/LATEST/policy.pt gemma-7b 0-4
LOSS_BETA=0.1
N_EPOCHS=10
EVAL_EVERY=192
EVAL_TRAIN_EVERY=192
NSEEDS=4
EVAL_ONLY_ONCE="False"
# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    --datasets)
      DATASETS="$2"
      shift # past argument
      shift # past value
      ;;
    --step_size)
      STEP_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --divide_by_totalcount)
      DIVIDE_BY_TOTALCOUNT="$2"
      shift # past argument
      shift # past value
      ;;
    --train_frac)
      TRAIN_FRAC="$2"
      shift # past argument
      shift # past value
      ;;
    --loss)
      LOSS="$2"
      shift # past argument
      shift # past value
      ;;
    --gradient_accumulation_steps)
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift # past argument
      shift # past value
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --eval_batch_size)
      EVAL_BATCH_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --sample_during_eval)
      SAMPLE_DURING_EVAL="$2"
      shift # past argument
      shift # past value
      ;;
    --trainer)
      TRAINER="$2"
      shift # past argument
      shift # past value
      ;;
    --lr)
      LR="$2"
      shift # past argument
      shift # past value
      ;;
    --model_archive)
      MODEL_ARCHIVE="$2"
      shift # past argument
      shift # past value
      ;;
    --loss_beta)
      LOSS_BETA="$2"
      shift # past argument
      shift # past value
      ;;
    --n_epochs)
      N_EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --eval_every)
      EVAL_EVERY="$2"
      shift # past argument
      shift # past value
      ;;
    --eval_train_every)
      EVAL_TRAIN_EVERY="$2"
      shift # past argument
      shift # past value
      ;;
    --eval_only_once)
      EVAL_ONLY_ONCE="$2"
      shift # past argument
      shift # past value
      ;;
    --nseeds)
      NSEEDS="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Loop over seeds 0 to nseeds
for SEED in $(seq 0 $NSEEDS)
do
  echo "Running training with seed $SEED"
  python -u train.py model=$MODEL datasets=[$DATASETS] train_frac=$TRAIN_FRAC loss=$LOSS gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE sample_during_eval=$SAMPLE_DURING_EVAL trainer=$TRAINER lr=$LR model.archive=$MODEL_ARCHIVE loss.beta=$LOSS_BETA seed=$SEED n_epochs=$N_EPOCHS eval_every=$EVAL_EVERY eval_train_every=$EVAL_TRAIN_EVERY eval_only_once=$EVAL_ONLY_ONCE
done
