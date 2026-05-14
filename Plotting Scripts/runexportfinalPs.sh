#!/usr/bin/bash
#SBATCH --account=[your account username]
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=impsandatmsexport
#SBATCH --partition=amilan
#SBATCH --output=output_%j.log
#SBATCH --constraint=ib
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mihu1229@colorado.edu

set -euo pipefail

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MCenv

python	exportfinalPstoexcel.py