#!/bin/bash

name=sim_mar
config_path=$HOME/MAR/scripts/config.json

src_path=$HOME/MAR/scripts
main_path=$src_path/main.py
working_dir_path=$HOME/$name

group_name="ku_00249"
join="oe"
resource_list="nodes=1:thinnode:ppn=10,mem=10gb,walltime=43200"