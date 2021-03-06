#!/usr/bin/env bash
#SBATCH -A fnl_lab
#SBATCH --mem-per-cpu 8G
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task 10
#SBATCH --output temp_debug_output.txt
#SBATCH --error temp_debug_error.txt

if [ $# -eq 0 ]; then
  echo "REQUIRED: PATH"
  exit
fi

T1WIMAGE="$1"
JLFFOLDER="$2"
SUBJECTID="$3"

python /home/users/moorlu/PycharmProjects/jlf/temp_debug1.py "$T1WIMAGE" "$JLFFOLDER" "$SUBJECTID"