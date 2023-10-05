#!/bin/bash
set -e

source config.sh

# Load modules
echo "loading pbs modules"
module unload .
module load tools
module load anaconda3/2023.03

# Setup CFI experiment
echo "creating working directories"

now=$(date +"%Y%m%d%H%M%S")
working_dir_path=$HOME/$now\_$name
log_path=$working_dir_path/logs
sim_setup_path=$working_dir_path/$name.csv

mkdir -p $working_dir_path
mkdir -p $working_dir_path/results
mkdir -p $log_path

echo "generating simulation setup from $config_path"
python3 setup_sims.py -c $config_path -o $sim_setup_path

# Get number of runs
n_configs=$(python3 -c "import pandas as pd; print(pd.read_csv('$sim_setup_path').shape[0])")

n_repeats=$(python3 -c "import json; print(json.load(open('$config_path', 'r'))['num_repeats'])")

echo "found $n_configs configs to run $n_repeats times. queuing $n_repeats jobs"
# Run each experiment
for ((i=0; i<n_repeats ;i++)); do
	qsub \
		-W group_list=$group_name \
		-A $group_name \
		-j $join \
		-o $log_path/$i.txt \
		-l $resource_list \
		-t 0-$((n_configs-1)) \
		-F "`pwd` $i $sim_setup_path" \
		`pwd`/call_single_sim.sh
done