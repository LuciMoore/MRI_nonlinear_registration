#!/usr/bin/env bash

if [ $# -eq 0 ]; then
  echo "REQUIRED: PATH"
  exit
fi

T1WIMAGE="$1"
JLFFOLDER="$2"
SUBJECTID="$3"

python /home/users/moorlu/PycharmProjects/jlf/temp_debug2.py "$T1WIMAGE" "$JLFFOLDER" "$SUBJECTID"