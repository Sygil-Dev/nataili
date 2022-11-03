#/bin/bash

# Run this script with "--fix" to automatically fix the issues which can be fixed

# exit script directly if any command fails
set -e

if [ "$1" == "--fix" ]
then
  echo "fix requested"
  BLACK_OPTS=""
  ISORT_OPTS=""
else
  echo "fix not requested"
  BLACK_OPTS="--check --diff"
  ISORT_OPTS="--check-only --diff"
fi

SRC="*.py nataili"

black $BLACK_OPTS $SRC
flake8 $SRC
isort $ISORT_OPTS $SRC
