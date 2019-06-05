#!/usr/bin/env bash
#SBATCH --mem-per-cpu 8G
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task 10
#SBATCH --output jlf_2chreg_2chjlf_output.txt
#SBATCH --error jlf_2chreg_2chjlf_error.txt

if [ $# -eq 0 ]; then
  echo "REQUIRED: PATH"
  exit
fi

T1WFOLDER="$1"
JLFFOLDER="$3"
SUBJECTID="$4"
NCPUS=$SLURM_CPUS_PER_TASK

python /home/users/moorlu/PycharmProjects/jlf/temp_2chreg_2chjlf.py "$T1WFOLDER" "$JLFFOLDER" "$SUBJECTID" --njobs "$NCPUS"