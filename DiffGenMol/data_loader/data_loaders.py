import deepchem as dc
import deepsmiles
from multiprocessing import cpu_count
import numpy as np
import os
import pandas as pd
import pickle
import selfies as sf
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import utils

# Class of selfies dataset
class DatasetSelfies(Dataset):
    def __init__(self, x: np.ndarray, y: Tensor, largest_selfie_len, selfies_alphabet, symbol_to_int):
        super().__init__()
        self.x = x
        self.y = y
        self.largest_selfie_len = largest_selfie_len
        self.selfies_alphabet = selfies_alphabet
        self.symbol_to_int = symbol_to_int

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'continous_vectors': utils.one_selfies_to_continous_mol(self.x[idx], self.largest_selfie_len, 
                                                                        self.symbol_to_int), 'classe': self.y[idx]}

# Selfies Loader from QM9 dataset
class QM9DataLoaderSelfies():
    def __init__(self, dataset_size, batch_size, num_classes, type_property, canonicalize_smiles, shuffle, config):
        self.config = config
        self.logger = config.get_logger('train')

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.type_property = type_property

        # Load smiles dataset
        if canonicalize_smiles:
            dataset_smiles_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_can_smiles.pickle')
        else:
            dataset_smiles_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_smiles.pickle')
        if os.path.isfile(dataset_smiles_pickle):
            self.logger.info('Loading existing smiles dataset')
            self.train_smiles =  pd.read_pickle(dataset_smiles_pickle)
        else:
            self.logger.info('Loading dataset')
            _, datasets, _ = dc.molnet.load_qm9(featurizer='ECFP')
            train_smiles_prep = pd.DataFrame(data={'smiles': datasets[0].ids})

            if canonicalize_smiles:
                self.logger.info('With Canonicalisation')
                # Canonicalize smiles
                train_smiles_prep['smiles'] = utils.canonicalize_smiles(train_smiles_prep['smiles'])
            else:
                self.logger.info('Without Canonicalisation')

            if dataset_size > 0:
                train_smiles_prep = train_smiles_prep[['smiles']][:dataset_size]
            else:
                train_smiles_prep = train_smiles_prep[['smiles']]

            self.train_smiles = train_smiles_prep['smiles']
            self.train_smiles.to_pickle(dataset_smiles_pickle)
            self.logger.info('Smiles dataset saved')
        
        self.logger.info(f'dataset_size: {len(self.train_smiles)}')
        self.logger.info(f'dataset_max_length: {self.train_smiles.str.len().max()}')
        assert utils.has_int_squareroot(len(self.train_smiles)), 'number of samples must have an integer square root'

        # Convert to Selfies
        if canonicalize_smiles:
            dataset_selfies_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_can_selfies.pickle')
        else:
            dataset_selfies_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_selfies.pickle')

        if os.path.isfile(dataset_selfies_pickle):
            self.logger.info('Loading existing selfies conversion')
            with open(dataset_selfies_pickle, 'rb') as f:
                self.train_selfies = pickle.load(f)
        else:
            self.logger.info('Converting to selfies')
            self.train_selfies = utils.get_valid_selfies(utils.smiles_to_selfies(self.train_smiles))
            with open(dataset_selfies_pickle, 'wb') as f:
                pickle.dump(self.train_selfies, f)
            self.logger.info('Selfies conversion saved')
        
        # Calculation Selfies features (alphabet and dico)
        self.logger.info('Calculate selfies features')
        self.largest_value_len, self.selfies_alphabet, self.symbol_to_int, self.int_mol = utils.get_selfies_features(self.train_selfies, canonicalize_smiles)

        self.seq_length = len(self.selfies_alphabet)
        self.logger.info(f'seq_length: {self.seq_length}')
        self.logger.info(f'channels: {self.largest_value_len}')
        self.nb_mols = len(self.train_selfies)


        # Calculation properties and classes
        if canonicalize_smiles:
            prop_weight_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_can_prop_weight.pickle')
            prop_logp_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_can_prop_logp.pickle')
            prop_qed_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_can_prop_qed.pickle')
            prop_sas_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_can_prop_sas.pickle')
            classes_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_can_classes_' + self.type_property + '.pickle')
            classes_breakpoints_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_can_classes_' + self.type_property + '_breakpoints.pickle')
        else:
            prop_weight_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_prop_weight.pickle')
            prop_logp_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_prop_logp.pickle')
            prop_qed_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_prop_qed.pickle')
            prop_sas_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_prop_sas.pickle')
            classes_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_classes_' + self.type_property + '.pickle')
            classes_breakpoints_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sf_classes_' + self.type_property + '_breakpoints.pickle')
        if os.path.isfile(prop_weight_pickle) and os.path.isfile(prop_logp_pickle) and os.path.isfile(prop_qed_pickle) and os.path.isfile(prop_sas_pickle) and os.path.isfile(classes_pickle) and os.path.isfile(classes_breakpoints_pickle):
            self.logger.info('Loading existing properties and classes')
            with open(prop_weight_pickle, 'rb') as f:
                self.prop_weight = pickle.load(f)
            with open(prop_logp_pickle, 'rb') as f:
                self.prop_logp = pickle.load(f)
            with open(prop_qed_pickle, 'rb') as f:
                self.prop_qed = pickle.load(f)
            with open(prop_sas_pickle, 'rb') as f:
                self.prop_sas = pickle.load(f)
            with open(classes_pickle, 'rb') as f:
                self.train_classes = pickle.load(f)
            with open(classes_breakpoints_pickle, 'rb') as f:
                self.classes_breakpoints = pickle.load(f)
        else:
            self.logger.info('Calculating and discretize molecules properties (classes)')
            self.prop_weight = utils.get_selfies_properties(self.train_selfies,"Weight")
            self.prop_logp = utils.get_selfies_properties(self.train_selfies,"LogP")
            self.prop_qed = utils.get_selfies_properties(self.train_selfies,"QED")
            self.prop_sas = utils.get_selfies_properties(self.train_selfies,"SAS")
            self.train_classes, self.classes_breakpoints = utils.discretize_continuous_values(self.get_train_prop(), self.num_classes)       
            with open(prop_weight_pickle, 'wb') as f:
                pickle.dump(self.prop_weight, f)
            with open(prop_logp_pickle, 'wb') as f:
                pickle.dump(self.prop_logp, f)
            with open(prop_qed_pickle, 'wb') as f:
                pickle.dump(self.prop_qed, f)
            with open(prop_sas_pickle, 'wb') as f:
                pickle.dump(self.prop_sas, f)
            with open(classes_pickle, 'wb') as f:
                pickle.dump(self.train_classes, f)
            with open(classes_breakpoints_pickle, 'wb') as f:
                pickle.dump(self.classes_breakpoints, f)
            d = {'smiles':self.train_smiles,'LogP':self.prop_logp,'QED':self.prop_qed,'SAS':self.prop_sas}
            df = pd.DataFrame(d)
            if canonicalize_smiles:
                df.to_csv(f'{self.config.result_dir}/train_smiles_can.csv', index=False)
            else:
                df.to_csv(f'{self.config.result_dir}/train_smiles.csv', index=False)
            self.logger.info('Properties and classes breakpoints saved')

        self.logger.info('Creating DatasetSelfies')
        self.dataset = DatasetSelfies(self.train_selfies, self.train_classes, self.largest_value_len, self.selfies_alphabet, self.symbol_to_int)
        # create dataloader
        self.logger.info('Creating DatasetLoader')
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, pin_memory = True, num_workers = cpu_count())
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_train_smiles(self):
        return self.train_smiles
    
    def get_nb_mols(self):
        return self.nb_mols
    
    def get_train_prop(self):
        if self.type_property == "Weight":
            self.prop_weight 
        elif self.type_property == "LogP":
            return self.prop_logp
        elif self.type_property == "QED":
            return self.prop_qed
        elif self.type_property == "SAS":
            return self.prop_sas 
    
    def get_prop_weight(self):
        return self.prop_weight
    
    def get_prop_logp(self):
        return self.prop_logp
    
    def get_prop_qed(self):
        return self.prop_qed
    
    def get_prop_sas(self):
        return self.prop_sas
    
    def get_train_classes(self):
        return self.train_classes

    def get_num_classes(self):
        return self.num_classes
    
    def get_dataloader(self):
        return self.dataloader
    
    def get_seq_length(self):
        return self.seq_length
    
    def get_type_property(self):
        return self.type_property
    
    def get_classes_breakpoints(self):
        return self.classes_breakpoints

    def get_selfies_alphabet(self):
        return self.selfies_alphabet   

    def get_largest_value_len(self):
        return self.largest_value_len 

    def get_int_mol(self):
        return self.int_mol

# Class of smiles dataset
class DatasetSmiles(Dataset):
    def __init__(self, x: np.ndarray, y: Tensor, featurizer):
        super().__init__()
        self.x = x
        self.y = y
        self.featurizer = featurizer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'continous_vectors': utils.one_smiles_to_continous_mol(self.x[idx], self.featurizer), 'classe': self.y[idx]}

# Selfies Loader from QM9 dataset
class QM9DataLoaderSmiles():
    def __init__(self, dataset_size, batch_size, num_classes, type_property, canonicalize_smiles, shuffle, config):
        self.config = config
        self.logger = config.get_logger('train')

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.type_property = type_property

        # Load smiles dataset
        if canonicalize_smiles:
            dataset_smiles_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_can_smiles.pickle')
        else:
            dataset_smiles_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_smiles.pickle')
        
        if os.path.isfile(dataset_smiles_pickle):
            print("Loading existing smiles dataset")
            self.train_smiles =  pd.read_pickle(dataset_smiles_pickle)
        else:
            self.logger.info('Loading dataset')
            _, datasets, _ = dc.molnet.load_qm9(featurizer='ECFP')
            train_smiles_prep = pd.DataFrame(data={'smiles': datasets[0].ids})

            if canonicalize_smiles:
                self.logger.info('With Canonicalisation')
                # Canonicalize smiles
                train_smiles_prep['smiles'] = utils.canonicalize_smiles(train_smiles_prep['smiles'])
            else:
                self.logger.info('Without Canonicalisation')

            if dataset_size > 0:
                train_smiles_prep = train_smiles_prep[['smiles']][:dataset_size]
            else:
                train_smiles_prep = train_smiles_prep[['smiles']]

            self.train_smiles = train_smiles_prep['smiles']
            self.train_smiles.to_pickle(dataset_smiles_pickle)
            if canonicalize_smiles:
                self.train_smiles.to_csv(f'{self.config.dataset_dir}/dataset_QM9_sm_can_train_smiles.csv', index=False)
            else:
                self.train_smiles.to_csv(f'{self.config.dataset_dir}/dataset_QM9_sm_train_smiles.csv', index=False)

            print("Smiles dataset saved")
        
        self.logger.info(f'dataset_size: {len(self.train_smiles)}')
        self.logger.info(f'dataset_max_length: {self.train_smiles.str.len().max()}')
        assert utils.has_int_squareroot(len(self.train_smiles)), 'number of samples must have an integer square root'
        
        # Preprocess and calculation Smiles features (alphabet)
        self.logger.info('Preprocess and calculate smiles features')
        self.deepsmiles_converter = deepsmiles.Converter(rings=True, branches=True)
        self.train_smiles_encoded, self.replace_dict, self.smiles_alphabet, self.largest_value_len = utils.preprocess_smiles(self.train_smiles, self.deepsmiles_converter)

        self.seq_length = len(self.smiles_alphabet)+1

        self.logger.info(f'seq_length: {self.seq_length}')
        self.logger.info(f'channels: {self.largest_value_len}')
        self.nb_mols = len(self.train_smiles_encoded)

        # Calculation properties and classes
        if canonicalize_smiles:
            prop_weight_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_can_prop_weight.pickle')
            prop_logp_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_can_prop_logp.pickle')
            prop_qed_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_can_prop_qed.pickle')
            prop_sas_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_can_prop_sas.pickle')
            classes_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_can_classes_' + self.type_property + '.pickle')
            classes_breakpoints_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_can_classes_' + self.type_property + '_breakpoints.pickle')
        else:
            prop_weight_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_prop_weight.pickle')
            prop_logp_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_prop_logp.pickle')
            prop_qed_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_prop_qed.pickle')
            prop_sas_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_prop_sas.pickle')
            classes_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_classes_' + self.type_property + '.pickle')
            classes_breakpoints_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_sm_classes_' + self.type_property + '_breakpoints.pickle')
        
        if os.path.isfile(prop_weight_pickle) and os.path.isfile(prop_logp_pickle) and os.path.isfile(prop_qed_pickle) and os.path.isfile(prop_sas_pickle) and os.path.isfile(classes_pickle) and os.path.isfile(classes_breakpoints_pickle):
            print("Loading existing properties and classes")
            with open(prop_weight_pickle, 'rb') as f:
                self.prop_weight = pickle.load(f)
            with open(prop_logp_pickle, 'rb') as f:
                self.prop_logp = pickle.load(f)
            with open(prop_qed_pickle, 'rb') as f:
                self.prop_qed = pickle.load(f)
            with open(prop_sas_pickle, 'rb') as f:
                self.prop_sas = pickle.load(f)
            with open(classes_pickle, 'rb') as f:
                self.train_classes = pickle.load(f)
            with open(classes_breakpoints_pickle, 'rb') as f:
                self.classes_breakpoints = pickle.load(f)
        else:
            self.logger.info('Calculating and discretize molecules properties (classes)')
            self.prop_weight = utils.get_smiles_properties(self.train_smiles,"Weight")
            self.prop_logp = utils.get_smiles_properties(self.train_smiles,"LogP")
            self.prop_qed = utils.get_smiles_properties(self.train_smiles,"QED")
            self.prop_sas = utils.get_smiles_properties(self.train_smiles,"SAS")
            self.train_classes, self.classes_breakpoints = utils.discretize_continuous_values(self.get_train_prop(), self.num_classes)       
            with open(prop_weight_pickle, 'wb') as f:
                pickle.dump(self.prop_weight, f)
            with open(prop_logp_pickle, 'wb') as f:
                pickle.dump(self.prop_logp, f)
            with open(prop_qed_pickle, 'wb') as f:
                pickle.dump(self.prop_qed, f)
            with open(prop_sas_pickle, 'wb') as f:
                pickle.dump(self.prop_sas, f)
            with open(classes_pickle, 'wb') as f:
                pickle.dump(self.train_classes, f)
            with open(classes_breakpoints_pickle, 'wb') as f:
                pickle.dump(self.classes_breakpoints, f)
            print("Properties and classes breakpoints saved")
 
        self.featurizer = dc.feat.OneHotFeaturizer(charset=self.smiles_alphabet, max_length=self.largest_value_len)
        self.logger.info('Creating DatasetSmiles')
        self.dataset = DatasetSmiles(self.train_smiles_encoded, self.train_classes, self.featurizer)
        # create dataloader
        self.logger.info('Creating DatasetLoader')
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, pin_memory = True, num_workers = cpu_count())
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_train_smiles(self):
        return self.train_smiles
    
    def get_nb_mols(self):
        return self.nb_mols
    
    def get_train_prop(self):
        if self.type_property == "Weight":
            self.prop_weight 
        elif self.type_property == "LogP":
            return self.prop_logp
        elif self.type_property == "QED":
            return self.prop_qed
        elif self.type_property == "SAS":
            return self.prop_sas 
    
    def get_prop_weight(self):
        return self.prop_weight
    
    def get_prop_logp(self):
        return self.prop_logp
    
    def get_prop_qed(self):
        return self.prop_qed
    
    def get_prop_sas(self):
        return self.prop_sas
    
    def get_train_classes(self):
        return self.train_classes

    def get_num_classes(self):
        return self.num_classes
    
    def get_dataloader(self):
        return self.dataloader
    
    def get_seq_length(self):
        return self.seq_length
    
    def get_type_property(self):
        return self.type_property
    
    def get_classes_breakpoints(self):
        return self.classes_breakpoints

    def get_smiles_alphabet(self):
        return self.smiles_alphabet   

    def get_largest_value_len(self):
        return self.largest_value_len
    
    def get_featurizer(self):
        return self.featurizer
    
    def get_replace_dict(self):
        return self.replace_dict
    
    def get_deepsmiles_converter(self):
        return self.deepsmiles_converter

