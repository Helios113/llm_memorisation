#!/bin/bash
#SBATCH --job-name=my_poetry_job
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G


# Add mauao specific commands here


# Navigate to the project directory
cd /path/to/your/project

# Run the script using poetry
poetry run python your_script.py
