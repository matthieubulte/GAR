import argparse
from collections.abc import Iterable
import itertools
import json
import pandas as pd

parser = argparse.ArgumentParser(
    description="xxxxx",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-c", "--config", type=str,
                    help="path to the config.json file", required=True)

parser.add_argument("-o", "--output", type=str,
                    help="path of where to place the sim config", required=True)

args = vars(parser.parse_args())

with open(args.pop('config'), 'r') as f: config = json.load(f)

# Remove num_repeats which isn't used by the tasks
del config['num_repeats']

keys, values = zip(*config.items())
values = [ v if isinstance(v, Iterable) else [v] for v in values]
sim_settings = [dict(zip(keys, p)) for p in itertools.product(*values)]

pd.DataFrame.from_dict(sim_settings).to_csv(args.pop("output"), index=False)