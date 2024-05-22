#!/bin/bash

# Default parameters
MODEL="gemma-2b"
DATASETS="goqa_0,goqa_1,goqa_2,goqa_3,goqa_4"
TRAIN_FRAC=0.8
LOSS="sft"
GRADIENT_ACCUMULATION_STEPS=2
BATCH_SIZE=16
EVAL_BATCH_SIZE=8
SAMPLE_DURING_EVAL="False"
TRAINER="GroupTrainer"
LR=1e-4
N_EPOCHS=1
EVAL_EVERY=192
EVAL_TRAIN_EVERY=192
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
    *)    # unknown option
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

python -u train.py model=$MODEL datasets=[$DATASETS] train_frac=$TRAIN_FRAC loss=$LOSS gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE sample_during_eval=$SAMPLE_DURING_EVAL trainer=$TRAINER lr=$LR eval_every=$EVAL_EVERY eval_train_every=$EVAL_TRAIN_EVERY