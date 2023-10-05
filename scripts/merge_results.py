import argparse
import pandas as pd
import pickle
import os

parser = argparse.ArgumentParser(
    description="xxxxx",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--input", type=str,
                    help="path to the directory in which results are stored", required=True)


parser.add_argument("-o", "--output", type=str,
                    help="path to where to put the merged results", required=True)

args = vars(parser.parse_args())
results_dir = args.pop('input')
target_file = args.pop('output')

results = []

for filename in os.listdir(results_dir):
    with open(f'{results_dir}/{filename}', 'rb') as f:
        res = pickle.load(f).to_dict()
        res.update(res['result'])
        del res['result']
        results.append(pd.Series(res))

df = pd.DataFrame(results)
with open(target_file, 'wb') as f:
    pickle.dump(df, f)