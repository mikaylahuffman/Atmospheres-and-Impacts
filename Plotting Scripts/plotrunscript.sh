#!/usr/bin/bash
#SBATCH --account=ucb637_asc1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --time=0:06:00
#SBATCH --job-name=impsandatmsanalysis
#SBATCH --partition=amilan
#SBATCH --output=output_%j.log
#SBATCH --constraint=ib
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mihu1229@colorado.edu


source activate base
conda activate MCenv

python plotting.py 