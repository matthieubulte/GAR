#!/bin/bash
set -o xtrace 
set -e

cd $1
source config.sh

# Move to home directory
cd $src_path

# Load modules
module load tools
module load anaconda3/2023.03

pip3 install --user --quiet -r ../requirements.txt

PYTHONPATH="." python3 $main_path --repeat-id=$2 --path=$3 --config-id=$PBS_ARRAYID