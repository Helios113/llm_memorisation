#!/bin/bash
#SBATCH -J LLM_finetune 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#! specify node
#SBATCH -w mauao
#SBATCH --output=slurm_out/%x.%j.ans
#SBATCH --error=slurm_out/err_%x.%j.ans
# poetry lock
# poetry install
export TOKENIZERS_PARALLELISM=false
HYDRA_FULL_ERROR=1 srun poetry run python train_fl.py
