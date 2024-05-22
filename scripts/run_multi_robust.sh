#!/bin/bash

# Default parameters
MODEL="gemma-2b"
DATASETS="goqa_0,goqa_1,goqa_2,goqa_3,goqa_4"
STEP_SIZES=("0.0000005") # exponential group rate
DIVIDE_BY_TOTALCOUNT="True"
TRAIN_FRAC=0.8
LOSS="ripo"
GRADIENT_ACCUMULATION_STEPS=2
BATCH_SIZE=16
EVAL_BATCH_SIZE=8
SAMPLE_DURING_EVAL="False"
TRAINER="GroupTrainerEarlyStop"
LR=6e-5
LABEL_SMOOTHING=0
MODEL_ARCHIVE="path-to-sft-policy"
LOSS_BETA=0.01
N_EPOCHS=30
EVAL_EVERY=960
EVAL_TRAIN_EVERY=192
EVAL_ONLY_ONCE="False"
NSEEDS=4
PATIENCE_FACTOR=2
SCHEDULER_METRIC="loss"
ADAPTIVE_STEP_SIZE="True"
USE_KFOLDSPLIT="False"
OPTIMIZER="AdamW"
STEP_FACTOR=0.5
MIN_LR=0.00000001
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
    --step_sizes)
      IFS=',' read -r -a STEP_SIZES <<< "$2"  # Read comma-separated list into array
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
    --scheduler_metric)
      SCHEDULER_METRIC="$2"
      shift # past argument
      shift # past value
      ;;
    --adaptive_step_size)
      ADAPTIVE_STEP_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --sample_during_eval)
      SAMPLE_DURING_EVAL="$2"
      shift # past argument
      shift # past value
      ;;
    --eval_only_once)
      EVAL_ONLY_ONCE="$2"
      shift # past argument
      shift # past value
      ;;
    --optimizer)
      OPTIMIZER="$2"
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
    --min_lr)
      MIN_LR="$2"
      shift # past argument
      shift # past value
      ;;
    --label_smoothing)
      LABEL_SMOOTHING="$2"
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
    --step_factor)
      STEP_FACTOR="$2"
      shift # past argument
      shift # past value
      ;;
    --n_epochs)
      N_EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --nseeds)
      NSEEDS="$2"
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
    --use_kfoldsplit)
      USE_KFOLDSPLIT="$2"
      shift # past argument
      shift # past value
      ;;
    --patience_factor)
      PATIENCE_FACTOR="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Outer loop over step sizes
for STEP_SIZE in "${STEP_SIZES[@]}"
do
  echo "Running simulations for step size $STEP_SIZE"
  # Loop over seeds 0 to 4
  for SEED in $(seq 0 $NSEEDS)
  do
    echo "Running training with seed $SEED"
    python -u train.py model=$MODEL min_lr=$MIN_LR loss.step_factor=$STEP_FACTOR datasets=[$DATASETS] optimizer=$OPTIMIZER loss.step_size=$STEP_SIZE use_kfoldsplit=$USE_KFOLDSPLIT patience_factor=$PATIENCE_FACTOR scheduler_metric=$SCHEDULER_METRIC loss.adaptive_step_size=$ADAPTIVE_STEP_SIZE loss.divide_by_totalcount=$DIVIDE_BY_TOTALCOUNT train_frac=$TRAIN_FRAC loss=$LOSS loss.label_smoothing=$LABEL_SMOOTHING gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE sample_during_eval=$SAMPLE_DURING_EVAL trainer=$TRAINER lr=$LR model.archive=$MODEL_ARCHIVE loss.beta=$LOSS_BETA seed=$SEED n_epochs=$N_EPOCHS eval_every=$EVAL_EVERY eval_train_every=$EVAL_TRAIN_EVERY eval_only_once=$EVAL_ONLY_ONCE
  done
done
