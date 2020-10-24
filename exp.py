from nnicli import Experiment
import argparse
import os
import sys
import yaml
import argparse
import nni


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='experiment config file')
    parser.add_argument('--port', type=int, default=8080, help='show webUI on which port')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    exp = Experiment()
    # exp.stop_experiment()
    exp.start_experiment(args.config_file, port=args.port)
