#!/bin/bash
#SBATCH -J LLM_finetune 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#! specify node
#SBATCH -w ngongotaha
#SBATCH --output=slurm_out/%x.%j.ans
#SBATCH --error=slurm_out/err_%x.%j.ans

# source  /nfs-share/dc912/miniconda3/envs/llm-memorization/.vevn/bin/activate
# HYDRA_FULL_ERROR=1 srun python train.py --config-name=config_apple
# --config-name=fl_config_pythia
#WANDB_MODE=disabled 
HYDRA_FULL_ERROR=1 srun poetry run python train_fl.py --config_name=centralize_config_pythia.yaml > logs/test-medical-centralize-new.log 2>&1
