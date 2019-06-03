#!/usr/bin/env bash
#SBATCH --mem-per-cpu 4G
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task 10
#SBATCH --output jlf1ch_output.txt
#SBATCH --error jlf1ch_error.txt

if [ $# -eq 0 ]; then
  echo "REQUIRED: PATH"
  exit
fi

T1WIMAGE="$1"
JLFFOLDER="$2"
SUBJECTID="$3"
NCPUS=$SLURM_CPUS_PER_TASK

python /home/users/moorlu/PycharmProjects/jlf/joint_label_fusion_1ch.py "$T1WIMAGE" "$JLFFOLDER" "$SUBJECTID" --njobs "$NCPUS"