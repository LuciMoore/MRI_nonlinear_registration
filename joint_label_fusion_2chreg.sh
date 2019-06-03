#!/bin/bash -e
#SBATCH --mem-per-cpu 4G
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task 10
#SBATCH --output jlf_2chreg_output.txt
#SBATCH --error jlf_2chreg_error.txt

if [ $# -eq 0 ]; then
  echo "REQUIRED: PATH"
  exit
fi

T1WIMAGE="$1"
T2WIMAGE="$2"
JLFFOLDER="$3"
SUBJECTID="$4"
NCPUS=$SLURM_CPUS_PER_TASK

python /home/users/moorlu/PycharmProjects/jlf/joint_label_fusion_2chreg.py "$T1WIMAGE" "$T2WIMAGE" "$JLFFOLDER" "$SUBJECTID" --njobs "$NCPUS"