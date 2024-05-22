# GRPO: Grup Robust Preference Optimization

This codebase builds upon the DPO codebase publicly available in github https://github.com/eric-mitchell/direct-preference-optimization 

## What is this repo?

This repo includes a reference implementation of the GRPO algorithm for training language models from preference data, as described in our paper


Similar to DPO, our pipeline has two stages:

1. Run supervised fine-tuning (SFT) on the dataset(s) of interest.
2. Run robust preference learning (IPO) on the model from step 1, using preference data.

The files in this repo are:
- `train.py`: the main entry point for training (either SFT/IPO/GRIPO preference-based training)
- `trainers.py`: the trainer classes (e.g., implementing the loop of learning)
- `utils.py`: some convenience functions used by multiple other files
- `preference_datasets.py`: dataset processing logic for both SFT and IPO/GRIPO preference-based training; 

In this codebase, we specifically use the Gemma-2b model and the configurations used are detailed in `config/model/gemma-2b.yaml`. To download and use the Gemma-2b model, kindly refer to https://huggingface.co/google/gemma-2b. It is a gated model, and hence requires access through huggingface. 

Our dataset is the global opinion data from https://huggingface.co/datasets/Anthropic/llm_global_opinions 

### Set up environment

First, create a virtualenv and install the dependencies. Python 3.10+ is recommended.

    python3 -m venv env
    source env/bin/activate
    pip install -r main_requirements.txt


In `config.yaml` setup your wandb details, so that results can be visualized there.

## Running SFT

    sh scripts/run_sft.sh

Please run this command to reproduce the SFT used in our setup.
## Running IPO/GRIPO

To run IPO, one requires the reference policy which is the path to the sft file. Please run the following command for IPO with `path-to-sft-file` replaced with your actual path to SFT trained policy

    sh scripts/run_multi.sh --model.archive path-to-sft-file

The exact configurations we used in our IPO training are already set in `sh scripts/run_multi.sh`

Similarly for GRIPO

    sh scripts/run_multi_robust.sh --model.archive path-to-sft-file

Note these commands were run on a machine with 1 40GB A100 GPU. Further, we are running single GPU training, using `GroupEarlyStopTrainer` which
reduces the learning rate if there is improvement in loss values after a certain number of iterations and is tunable.

## Plotting results
In order to visualize the results, we collect data directly from wandb and plot the same. We include plotting scripts in `plot_scripts` folder that performs this. Kindly change the wandb details and `path-to-sft-file` in the plot scripts to retrieve the plots.

`plot_scripts/plot_from_wandb_full_metrics.py` plots all the relevant metrics tracked in our experiments
`plot_scripts/plot_from_wandb_paper_plots.py` reproduces the plots mentioned in the paper




