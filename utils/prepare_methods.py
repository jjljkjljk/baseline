import pickle
import pandas as pd
import logging
from mlp_retrosyn.mlp_inference import MLPModel

import json
def prepare_starting_molecules(filename):
    logging.info('Loading building blocks from %s' % filename)

    if filename[-3:] == 'csv':
        if 'chembl' in filename:
            starting_mols = set(list(pd.read_csv(filename, sep=";")['Smiles']))
        else:
            starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)
    logging.info('totally %d building blocks loaded' % len(starting_mols))
    return starting_mols

#
def prepare_reaction_data(filename):
    logging.info('Loading reaction data from %s' % filename)
    f = open(filename, 'r')
    reaction_data = f.read()
    reaction_data_dict = json.loads(reaction_data)
    logging.info('%d reaction data loaded' % len(reaction_data_dict))
    return reaction_data_dict


def prepare_mlp(templates, model_dump):
    logging.info('Templates: %s' % templates)
    logging.info('Loading one-step-retrosynthetic model from %s' % model_dump)
    one_step = MLPModel(model_dump, templates, device=-1)
    return one_step


