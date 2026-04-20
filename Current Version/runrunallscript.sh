#!/usr/bin/bash
#SBATCH --account=ucb637_asc1
#SBATCH --qos=long
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --job-name=impactsatmospheres
#SBATCH --partition=amilan
#SBATCH --output=output_%j.log
#SBATCH --constraint=ib
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mihu1229@colorado.edu
#SBATCH --array=0-17

source activate base
conda activate MCenv

listplanets=('Earth' 'Earth' 'Earth' 'Earth' 'Earth' 'Earth' 'Mars' 'Mars' 'Mars' 'Mars' 'Mars' 'Mars' 'Venus' 'Venus' 'Venus' 'Venus' 'Venus' 'Venus')
listpressures=(0.006 0.1 0.25 1 10 92.5 0.006 0.1 0.25 1 10 92.5 0.006 0.1 0.25 1 10  92.5)

python v21.py --planet ${listplanets[$SLURM_ARRAY_TASK_ID]} --startingP ${listpressures[$SLURM_ARRAY_TASK_ID]} 