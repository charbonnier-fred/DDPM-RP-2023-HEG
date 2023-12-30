# Adapted from Victor Huang's GitHub (victoresque, pytorch-template)
import argparse
import collections
import numpy as np
import torch
from data_loader import data_loaders
from trainer import trainers
from parse_config import ConfigParser

# Random seeds for reproducibility
SEED = 2807
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    logger.info('###############')
    logger.info('DiffGenMol V1.0')
    logger.info('###############')
    logger.info('Authors : Charbonnier Frédéric and Clerc Joël, 2023')
    logger.info('Licence : CC BY')
    logger.info('Based on: Phil Wang''s GitHub (lucidrains, diffusion 1D) and Victor Huang''s GitHub (victoresque, pytorch-template)')
    logger.info('Comments: Created as part of the Master in Information Sciences research project "Learning to generate molecules"')
    logger.info('at the Haute école de gestion de Genève')
    logger.info('###############')
    # setup data_loader instance
    data_loader = config.init_obj('data_loader', data_loaders, config=config)
    logger.info(data_loader)

    # path of best model
    best_model_path = './saved/models/DiffGenMol-Selfies-Can-LogP/1121_024439/best-model.pt'
    
    # setup trainer instance
    trainer = config.init_obj('trainer', trainers, config=config, data_loader=data_loader, best_model_path=best_model_path)
    logger.info(trainer)

    # evaluate best model
    trainer.evaluate_best_model()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='DiffGenMol')
    # config json
    args.add_argument('-c', '--config', default="config_selfies_can_logp_heavy.json", type=str,
                      help='config file path (default: None)')
    # custom cli options
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='trainer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
