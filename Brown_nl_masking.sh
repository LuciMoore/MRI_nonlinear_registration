#!/bin/bash -e
##SBATCH --mem-per-cpu 4G
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task 10
#SBATCH --output nlmaskingoutput.txt
#SBATCH --error nlmaskingerror.txt

if [ $# -eq 0 ]; then
  echo "REQUIRED: PATH"
  exit
fi

FOLDER="$1"
NCPUS=$SLURM_CPUS_PER_TASK

python /home/users/moorlu/PycharmProjects/jlf/Brown_nl_masking.py "$FOLDER" --njobs "$NCPUS"
