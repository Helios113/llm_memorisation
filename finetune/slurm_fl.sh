#!/bin/bash
#SBATCH -J LLM_finetune 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#! specify node
#SBATCH -w mauao
#SBATCH --output=slurm_out/%x.%j.ans
#SBATCH --error=slurm_out/err_%x.%j.ans
pyenv local 3.11
# # Check if the Poetry environment is installed, if not, create it
# if ! poetry env info &> /dev/null
# then
#     echo "Poetry environment could not be found. Creating Poetry environment..."
#     poetry lock    
#     poetry install
# fi

# source /nfs-share/pa511/developments/lm_memorisation/.vevn/bin/activate
# HYDRA_FULL_ERROR=1 srun python train.py --config-name=config_apple
# --config-name=fl_config_pythia
#WANDB_MODE=disabled 
poetry lock    
poetry install
HYDRA_FULL_ERROR=1 srun poetry run python train_fl.py
