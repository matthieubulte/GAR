import argparse
import os
import pandas as pd
import pathlib

parser = argparse.ArgumentParser(
    description="Aggregate simulation results",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("name", type=str,
                    help="name of the run")
args = vars(parser.parse_args())

name = args.pop("name")
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")

all_files_in_results = os.listdir(results_dir)
result_paths_in_results_dir = [os.path.join(results_dir, filename) for filename in all_files_in_results if filename.rsplit(".", 1)[0].rsplit("_", 1)[0] == name]
loaded_results = [pd.read_pickle(path) for path in result_paths_in_results_dir]

final_df = pd.concat(loaded_results, axis=1).T.sort_index()
final_df.to_pickle(os.path.join(script_dir, "clean_results", name + ".pkl"))


setup_path = os.path.join(results_dir, name + ".pkl")
pathlib.Path(setup_path).unlink(missing_ok=True)

for path in result_paths_in_results_dir:
    pathlib.Path(path).unlink(missing_ok=True)