#!/usr/bin/bash
#SBATCH --account=ucb637_asc1
#SBATCH --qos=long
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --job-name=impactsatmospheres
#SBATCH --output=output_%A_%a.log
#SBATCH --constraint=ib
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mihu1229@colorado.edu
#SBATCH --array=0-17

source activate base
conda activate MCenv

listplanets=('Earth' 'Earth' 'Earth' 'Earth' 'Earth' 'Earth' 'Mars' 'Mars' 'Mars' 'Mars' 'Mars' 'Mars' 'Venus' 'Venus' 'Venus' 'Venus' 'Venus' 'Venus')
listpressures=(0.006 0.1 0.25 1 10 92.5 0.006 0.1 0.25 1 10 92.5 0.006 0.1 0.25 1 10  92.5)

export SLURM_NTASKS=${SLURM_CPUS_PER_TASK}

echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Planet: ${listplanets[$SLURM_ARRAY_TASK_ID]}"
echo "Starting pressure: ${listpressures[$SLURM_ARRAY_TASK_ID]}"
echo "Using ${SLURM_NTASKS} worker processes"

python v22.py --planet "${listplanets[$SLURM_ARRAY_TASK_ID]}" --startingP "${listpressures[$SLURM_ARRAY_TASK_ID]}"