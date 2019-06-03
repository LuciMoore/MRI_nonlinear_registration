#!/usr/bin/env bash
##SBATCH --mem-per-cpu 4G
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task 10
#SBATCH --output nlreg_output.txt
#SBATCH --error nlreg_error.txt

if [ $# -eq 0 ]; then
  echo "REQUIRED: PATH"
  exit
fi

FOLDER="$1"
NCPUS=$SLURM_CPUS_PER_TASK

python /home/users/moorlu/PycharmProjects/jlf/nonlinear_reg.py "$FOLDER" --njobs "$NCPUS"