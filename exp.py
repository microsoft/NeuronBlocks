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


def get_hyperparameters(conf):
    parameters = nni.get_next_parameter()
    conf["training_params"]["optimizer"]["params"]["lr"] = parameters["learning_rate"]
    conf['architecture'][3]['conf']['dropout'] = parameters['LSTM_dropout_rate']
    conf['architecture'][4]['conf']['dropout'] = parameters['LSTM_dropout_rate']
    conf['training_params']['batch_size'] = parameters['batch_size']
    conf['architecture'][0]['conf']['word']['dropout'] = parameters['embedding_drop']
    conf['training_params']['lr_decay'] = parameters['decay']
    conf['loss']['losses'][0]['conf']['weight'] = [parameters['loss_weight'], 1 - parameters['loss_weight']]
    conf['architecture'][2]['conf']['dropout'] = parameters['query_drop']
    conf['architecture'][3]['conf']['dropout'] = parameters['passage_drop']
    return conf
