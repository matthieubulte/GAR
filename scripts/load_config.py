import pandas as pd
import argparse

def load_config():
    import pathlib, os

    parser = argparse.ArgumentParser(
        description="Run continuous semiparametric sim using setup file provided",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-c", "--config-id", type=int,
                        help="row index in the setup data frame to use for the simulation", required=True)
    parser.add_argument("-i", "--repeat-id", type=int,
                        help="id of the repeatition given the config id", required=True)
    parser.add_argument("-p", "--path", type=str,
                        help="path of the setup file", required=True)
    
    args = vars(parser.parse_args())

    setup_path = args.pop("path")
    name = pathlib.Path(setup_path).stem
    config_id = int(args.pop("config_id"))
    repeat_id = int(args.pop("repeat_id"))

    out_name = os.path.join(os.path.dirname(setup_path), "results", f"{name}_{config_id}_{repeat_id}.csv")

    return out_name, pd.read_csv(setup_path).iloc[config_id]
    

