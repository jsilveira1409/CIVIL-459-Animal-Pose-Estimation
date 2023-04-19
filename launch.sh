#!/bin/bash
#SBATCH --workdir /scratch/jsilveir
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos dlav
#SBATCH --account civil-459
#SBATCH --reservation civil-459

source venvs/jsilveira-venv/bin/activate
python3 main.py