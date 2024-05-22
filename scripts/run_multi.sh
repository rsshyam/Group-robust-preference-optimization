#!/bin/bash

# Default parameters
MODEL="gemma-2b"
DATASETS="goqa_0,goqa_1,goqa_2,goqa_3,goqa_4"
TRAIN_FRAC=0.8
LOSS="ipo" # ipo, ripo, rdpo
GRADIENT_ACCUMULATION_STEPS=2
BATCH_SIZE=16
EVAL_BATCH_SIZE=8
SAMPLE_DURING_EVAL="False"
TRAINER="GroupTrainerEarlyStop"
LR=2e-5
LABEL_SMOOTHING=0
MODEL_ARCHIVE="path-to-sft-policy"
LOSS_BETA=0.01
N_EPOCHS=30
EVAL_EVERY=960
EVAL_TRAIN_EVERY=192
NSEEDS=4
STSEED=0
EVAL_ONLY_ONCE="False"
PATIENCE_FACTOR=2
SCHEDULER_METRIC="loss"
USE_KFOLDSPLIT="False"
OPTIMIZER="AdamW"
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
    --min_lr)
      MIN_LR="$2"
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
    --scheduler_metric)
      SCHEDULER_METRIC="$2"
      shift # past argument
      shift # past value
      ;;
    --lr)
      LR="$2"
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
    --optimizer)
      OPTIMIZER="$2"
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
    --use_kfoldsplit)
      USE_KFOLDSPLIT="$2"
      shift # past argument
      shift # past value
      ;;
    --nseeds)
      NSEEDS="$2"
      shift # past argument
      shift # past value
      ;;
    --stseed)
      STSEED="$2"
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

# Loop over seeds 0 to nseeds
for SEED in $(seq $STSEED $NSEEDS)
do
  echo "Running training with seed $SEED"
  python -u train.py model=$MODEL min_lr=$MIN_LR optimizer=$OPTIMIZER datasets=[$DATASETS] use_kfoldsplit=$USE_KFOLDSPLIT train_frac=$TRAIN_FRAC patience_factor=$PATIENCE_FACTOR scheduler_metric=$SCHEDULER_METRIC loss=$LOSS gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE sample_during_eval=$SAMPLE_DURING_EVAL trainer=$TRAINER lr=$LR model.archive=$MODEL_ARCHIVE loss.beta=$LOSS_BETA seed=$SEED n_epochs=$N_EPOCHS eval_every=$EVAL_EVERY eval_train_every=$EVAL_TRAIN_EVERY eval_only_once=$EVAL_ONLY_ONCE loss.label_smoothing=$LABEL_SMOOTHING
done
