#!/bin/bash
set -o xtrace 
set -e

source $HOME/sim_config.sh

# Move to home directory
cd $src_path

# Load modules
module load tools
module load anaconda3/2023.03

pip3 install --user --quiet -r requirements.txt

PYTHONPATH="." python3 $main_path --repeat-id=$1 --path=$2 --config-id=$PBS_ARRAYID