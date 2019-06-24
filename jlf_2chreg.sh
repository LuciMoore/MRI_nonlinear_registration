#!/usr/bin/env bash
#SBATCH -A fnl_lab
#SBATCH --mem-per-cpu 8G
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task 10
#SBATCH --output jlf_2chreg_output.txt
#SBATCH --error jlf_2chreg_error.txt

if [ $# -eq 0 ]; then
  echo "REQUIRED: PATH"
  exit
fi

T1WFOLDER="$1"
JLFFOLDER="$2"
SUBJECTID="$3"
NCPUS=$SLURM_CPUS_PER_TASK

python /home/users/moorlu/PycharmProjects/jlf/jlf_2chreg.py "$T1WFOLDER" "$JLFFOLDER" "$SUBJECTID" --njobs "$NCPUS"