import os
import yaml
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file', default='configs/config_MedQA.yaml')
parser.add_argument('--method', type=str, help='method name')
parser.add_argument('--shift', type=str, help='shift type')
args = parser.parse_args()
config_path = args.config

with open(config_path) as f:
	cfg = yaml.load(f, Loader=yaml.Loader)


if args.method is not None:
	cfg['Experiment']['method'] = args.method

if args.shift is not None:
	cfg['Experiment']['shift']['type'] = args.shift


current_time = datetime.now()
timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_PATH = os.path.join(cfg["results_saved_base_path"], f"{timestamp_str}-{cfg['Experiment']['method']}")

os.mkdir(RESULTS_PATH)

