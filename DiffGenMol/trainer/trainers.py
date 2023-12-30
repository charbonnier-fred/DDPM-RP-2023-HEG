# Adapted from Phil Wang''s GitHub (lucidrains, diffusion 1D)
from accelerate import Accelerator
from ema_pytorch import EMA
import torch
import utils
from diffusion.diffusion_1d import Diffusion1D
from logger import TensorboardWriter
from model.unet_1d import Unet1D
from model import metrics
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm.auto import tqdm

# helpers
def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# trainer

class Trainer1D(object):
    def __init__(
        self,
        data_loader, 
        timesteps, 
        unet_dim, 
        unet_channels, 
        unet_cond_drop_prob,
        objective,
        beta_schedule, 
        tensorboard,
        config,
        *,
        model = None,
        best_model_path = None,
        gradient_accumulate_every = 2,
        train_lr = 1e-4,
        train_num_steps = 50000,
        ema_update_every = 363,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        eval_and_sample_every = 1000,
        num_samples = 20,
        num_samples_final_eval = 40,
        amp = True,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.
    ):
        super().__init__()

        self.config = config
        self.logger = config.get_logger('train')

        self.data_loader = data_loader

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        if model is None:
            unet = Unet1D(
                        dim = unet_dim,
                        dim_mults = (1, 2, 4, 8),
                        num_classes = self.data_loader.get_num_classes(),
                        channels = self.data_loader.get_largest_value_len(),
                        cond_drop_prob = unet_cond_drop_prob
                        )
            
            self.model = Diffusion1D(
                unet,
                seq_length = self.data_loader.get_seq_length(),
                timesteps = timesteps,
                objective = objective,
                beta_schedule = beta_schedule
            )
        else:
            self.model = model

        self.channels = self.data_loader.get_num_classes()

        # sampling and training hyperparameters

        self.num_samples = num_samples
        self.num_samples_final_eval = num_samples_final_eval
        self.eval_and_sample_every = eval_and_sample_every
        self.best_accuracy_match_classes = 0

        self.batch_size = self.data_loader.get_batch_size()
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = self.accelerator.prepare(self.data_loader.get_dataloader())
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(self.model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.result_dir = self.config.result_dir
        self.model_dir = self.config.model_dir
        self.log_dir = self.config.log_dir

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # best model if re-evaluate
        self.best_model_path = best_model_path

        # setup visualization writer instance                
        self.writer = TensorboardWriter(self.log_dir, self.logger, tensorboard)
        self.train_metrics = utils.MetricTracker('0_loss', 'A_UC_01_syntax_validity_samples', 'A_UC_02_mol_validity_samples',
                                                 'A_UC_03_uniqueness_samples', 'A_UC_04_novelty_samples','A_UC_05_KLdiv_samples',
                                                 'A_UC_06a_distr_weight_train', 'A_UC_06b_distr_weight_samples','A_UC_06c_wasserstein_distance_weight',
                                                 'A_UC_07a_distr_logp_train', 'A_UC_07b_distr_logp_samples','A_UC_07c_wasserstein_distance_logp',
                                                 'A_UC_08a_distr_qed_train', 'A_UC_08b_distr_qed_samples','A_UC_08c_wasserstein_distance_qed',
                                                 'A_UC_09a_distr_sas_train', 'A_UC_09b_distr_sas_samples','A_UC_09c_wasserstein_distance_sas',
                                                 'A_UC_10_wasserstein_distance_sum_props',
                                                 'B_C_01_syntax_validity_samples', 'B_C_02_mol_validity_samples',
                                                 'B_C_03_uniqueness_samples','B_C_04_novelty_samples','B_C_05_KLdiv_samples',
                                                 'B_C_06a_distr_prop_train', 'B_C_06b_distr_prop_samples','B_C_06c_wasserstein_distance_prop',
                                                 'B_C_07a_distr_classe_1_train', 'B_C_07b_distr_classe_1_samples','B_C_07c_wasserstein_distance_classe_1',
                                                 'B_C_08a_distr_classe_2_train', 'B_C_08b_distr_classe_2_samples','B_C_08c_wasserstein_distance_classe_2', 
                                                 'B_C_09a_distr_classe_3_train', 'B_C_09b_distr_classe_3_samples','B_C_09c_wasserstein_distance_classe_3',
                                                 'B_C_10_wasserstein_distance_sum_classes',
                                                 'B_C_11_classes_match', writer=self.writer)

    @property
    def device(self):
        return self.accelerator.device

    # Save best model
    def save_best_model(self):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': '1.0'
        }

        torch.save(data, str(self.model_dir / f'best-model.pt'))

    # Load best model
    def load_best_model(self):
        accelerator = self.accelerator
        device = accelerator.device

        if self.best_model_path is None:
          data = torch.load(str(self.model_dir / f'best-model.pt'), map_location=device)
        else:
          data = torch.load(str(self.best_model_path), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # Train model
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):

                    data = next(self.dl)
                    train_value = data['continous_vectors'].to(device) # normalized from 0 to 1
                    train_classe = data['classe'].to(device)    # say 10 classes

                    with self.accelerator.autocast():
                        loss = self.model(train_value, classes = train_classe)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')
                self.writer.set_step(self.step)
                self.train_metrics.update('0_loss', total_loss)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    # Evaluation and sampling
                    if self.step == 100 or self.step % self.eval_and_sample_every == 0:
                        with torch.no_grad():
                            accuracy_match_classes = self.eval_and_sample(self.num_samples, self.step)
                            result = self.train_metrics.result()
                            # save logged informations into log dict
                            log = {'step': self.step}
                            log.update(result)
                            # print logged informations to the screen
                            for key, value in log.items():
                                self.logger.info('    {:15s}: {}'.format(str(key), value))
                        # Save best conditional model
                        if self.step >= (self.train_num_steps/2) and self.best_accuracy_match_classes < accuracy_match_classes:
                            self.save_best_model()
                            self.best_accuracy_match_classes = accuracy_match_classes

                pbar.update(1)

        accelerator.print('training complete')
        # Final results with best model
        self.evaluate_best_model()
    
    # Evaluate best model
    def evaluate_best_model(self):
      self.load_best_model()
      self.train_metrics.reset()
      self.eval_and_sample(self.num_samples_final_eval, 'best')
      result = self.train_metrics.result()
      # save logged informations into log dict
      log = {'step': self.step}
      log.update(result)
      # print logged informations to the screen
      for key, value in log.items():
        self.logger.info('    {:15s}: {}'.format(str(key), value))
    
    # Evaluate and sample generated molecules
    def eval_and_sample(self, num_samples, step):
        # unconditional
        _num_samples = (num_samples // self.data_loader.get_num_classes()) * self.data_loader.get_num_classes()
        samples_classes_raw = torch.tensor([i // (_num_samples//self.data_loader.get_num_classes()) for i in range(_num_samples)]).to(self.device)
        samples_continous_mols_raw = self.ema.ema_model.sample(samples_classes_raw, cond_scale = 0)
        # QM9DataLoaderSelfies
        if self.data_loader.__class__.__name__ == 'QM9DataLoaderSelfies':
          samples_selfies_raw = utils.continous_mols_to_selfies(samples_continous_mols_raw, self.data_loader.get_selfies_alphabet(), self.data_loader.get_int_mol())
          samples_valid_selfies = utils.get_valid_selfies(samples_selfies_raw)
          accuracy_syntax_validity = metrics.accuracy_syntax_validity(samples_valid_selfies, samples_selfies_raw)
          samples_valid_mols, _, _ = utils.selfies_to_mols(samples_valid_selfies)
          samples_valid_smiles = utils.mols_to_smiles(samples_valid_mols)
        # QM9DataLoaderSmiles
        else:
          samples_smiles_raw = utils.continous_mols_to_smiles(samples_continous_mols_raw, self.data_loader.get_featurizer(), self.data_loader.get_replace_dict(), self.data_loader.get_deepsmiles_converter())
          samples_valid_smiles = utils.get_valid_smiles(samples_smiles_raw)
          accuracy_syntax_validity = metrics.accuracy_syntax_validity(samples_valid_smiles, samples_smiles_raw)
          samples_valid_mols, _, _ = utils.smiles_to_mols(samples_valid_smiles)

        samples_valid_smiles = utils.canonicalize_smiles(samples_valid_smiles)

        # metrics
        self.train_metrics.update('A_UC_01_syntax_validity_samples', accuracy_syntax_validity)

        # at least 1 valid molecule
        if accuracy_syntax_validity > 0:
          try:
            self.train_metrics.update('A_UC_02_mol_validity_samples', metrics.validity_score(samples_valid_smiles))
            self.train_metrics.update('A_UC_03_uniqueness_samples', metrics.uniqueness_score(samples_valid_smiles))
            self.train_metrics.update('A_UC_04_novelty_samples', metrics.novelty_score(samples_valid_smiles, self.data_loader.get_train_smiles()))
            self.train_metrics.update('A_UC_05_KLdiv_samples', metrics.KLdiv_score(samples_valid_smiles, self.data_loader.get_train_smiles()))
          except Exception:
            pass
          
          try:
            self.writer.add_histogram('A_UC_06b_distr_weight_samples',torch.FloatTensor(utils.get_smiles_properties(samples_valid_smiles,'Weight')))
            w_distance_weight = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_valid_smiles,'Weight'), self.data_loader.get_prop_weight())
            self.train_metrics.update('A_UC_06c_wasserstein_distance_weight', w_distance_weight)
            self.writer.add_histogram('A_UC_07b_distr_logp_samples',torch.FloatTensor(utils.get_smiles_properties(samples_valid_smiles,'LogP')))
            w_distance_logp = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_valid_smiles,'LogP'), self.data_loader.get_prop_logp())
            self.train_metrics.update('A_UC_07c_wasserstein_distance_logp', w_distance_logp)
            self.writer.add_histogram('A_UC_08b_distr_qed_samples',torch.FloatTensor(utils.get_smiles_properties(samples_valid_smiles,'QED')))
            w_distance_qed = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_valid_smiles,'QED'), self.data_loader.get_prop_qed())
            self.train_metrics.update('A_UC_08c_wasserstein_distance_qed', w_distance_qed)
            self.writer.add_histogram('A_UC_09b_distr_sas_samples',torch.FloatTensor(utils.get_smiles_properties(samples_valid_smiles,'SAS')))
            w_distance_sas = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_valid_smiles,'SAS'), self.data_loader.get_prop_sas())
            self.train_metrics.update('A_UC_09c_wasserstein_distance_sas',w_distance_sas)
            self.train_metrics.update('A_UC_10_wasserstein_distance_sum_props',w_distance_weight+w_distance_logp+w_distance_qed+w_distance_sas)      
          except Exception:
            pass
          
          # image
          try:
            self.writer.add_image('A_UC_11_mol_sample', transforms.ToTensor()(Chem.Draw.MolToImage(samples_valid_mols[0],size=(300,300))))
          except Exception:
            pass
          
        # only first eval
        if self.step == self.eval_and_sample_every:
            self.writer.add_histogram('A_UC_06a_distr_weight_train',torch.FloatTensor(self.data_loader.get_prop_weight()))
            self.writer.add_histogram('A_UC_07a_distr_logp_train',torch.FloatTensor(self.data_loader.get_prop_logp()))
            self.writer.add_histogram('A_UC_08a_distr_qed_train',torch.FloatTensor(self.data_loader.get_prop_qed()))
            self.writer.add_histogram('A_UC_09a_distr_sas_train',torch.FloatTensor(self.data_loader.get_prop_sas()))

        # sample
        # QM9DataLoaderSelfies
        if self.data_loader.__class__.__name__ == 'QM9DataLoaderSelfies':
           df = pd.DataFrame(samples_selfies_raw, columns=["selfies"])
           df.to_csv(f'{self.result_dir}/selfies-uc-{step}.csv', index=False)
           df = pd.DataFrame(samples_valid_smiles, columns=["smiles"])
        # QM9DataLoaderSmiles
        else:
           df = pd.DataFrame(samples_smiles_raw, columns=["smiles"])
           
        df.to_csv(f'{self.result_dir}/smiles-uc-{step}.csv', index=False)

        # conditional
        _num_samples = (num_samples // self.data_loader.get_num_classes()) * self.data_loader.get_num_classes()
        samples_classes_raw = torch.tensor([i // (_num_samples//self.data_loader.get_num_classes()) for i in range(_num_samples)]).to(self.device)
        samples_continous_mols_raw = self.ema.ema_model.sample(samples_classes_raw, cond_scale = 6.)
        # QM9DataLoaderSelfies
        if self.data_loader.__class__.__name__ == 'QM9DataLoaderSelfies':
          samples_selfies_raw = utils.continous_mols_to_selfies(samples_continous_mols_raw, self.data_loader.get_selfies_alphabet(), self.data_loader.get_int_mol())
          samples_valid_selfies, samples_valid_classes = utils.get_valid_selfies_and_classes(samples_selfies_raw, samples_classes_raw)
          accuracy_syntax_validity = metrics.accuracy_syntax_validity(samples_valid_selfies, samples_selfies_raw)
          samples_valid_mols, _, _ = utils.selfies_to_mols(samples_valid_selfies)
          samples_valid_smiles = utils.mols_to_smiles(samples_valid_mols)
        # QM9DataLoaderSmiles
        else:
          samples_smiles_raw = utils.continous_mols_to_smiles(samples_continous_mols_raw, self.data_loader.get_featurizer(), self.data_loader.get_replace_dict(), self.data_loader.get_deepsmiles_converter())
          samples_valid_smiles, samples_valid_classes = utils.get_valid_smiles_and_classes(samples_smiles_raw, samples_classes_raw)
          accuracy_syntax_validity = metrics.accuracy_syntax_validity(samples_valid_smiles, samples_smiles_raw)
          samples_valid_mols, _, _ = utils.smiles_to_mols(samples_valid_smiles)

        samples_valid_smiles = utils.canonicalize_smiles(samples_valid_smiles)
        
        # metrics
        self.train_metrics.update('B_C_01_syntax_validity_samples', accuracy_syntax_validity)

        # at least 1 valid molecule
        if accuracy_syntax_validity > 0:
          try:
            self.train_metrics.update('B_C_02_mol_validity_samples', metrics.validity_score(samples_valid_smiles))
            self.train_metrics.update('B_C_03_uniqueness_samples', metrics.uniqueness_score(samples_valid_smiles))
            self.train_metrics.update('B_C_04_novelty_samples', metrics.novelty_score(samples_valid_smiles, self.data_loader.get_train_smiles()))
            self.train_metrics.update('B_C_05_KLdiv_samples', metrics.KLdiv_score(samples_valid_smiles, self.data_loader.get_train_smiles()))
          except Exception:
            pass

          try:      
            prop_samples = utils.get_smiles_properties(samples_valid_smiles,self.data_loader.get_type_property())
            self.writer.add_histogram('B_C_06b_distr_prop_samples',torch.FloatTensor(prop_samples))
            w_distance = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_valid_smiles,self.data_loader.get_type_property()), self.data_loader.get_train_prop())
            self.train_metrics.update('B_C_06c_wasserstein_distance_prop', w_distance)
          except Exception:
            pass

          try:
            self.writer.add_histogram('B_C_07b_distr_classe_1_samples',torch.FloatTensor(utils.get_smiles_properties(utils.get_values_by_classe(samples_valid_smiles, samples_valid_classes, 0),self.data_loader.get_type_property())))
            classe_1_w_distance = metrics.wasserstein_distance_score(utils.get_smiles_properties(utils.get_values_by_classe(samples_valid_smiles, samples_valid_classes, 0),self.data_loader.get_type_property()), utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 0))
            self.train_metrics.update('B_C_07c_wasserstein_distance_classe_1', classe_1_w_distance)
            self.writer.add_histogram('B_C_08b_distr_classe_2_samples',torch.FloatTensor(utils.get_smiles_properties(utils.get_values_by_classe(samples_valid_smiles, samples_valid_classes, 1),self.data_loader.get_type_property())))
            classe_2_w_distance = metrics.wasserstein_distance_score(utils.get_smiles_properties(utils.get_values_by_classe(samples_valid_smiles, samples_valid_classes, 1),self.data_loader.get_type_property()), utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 1))
            self.train_metrics.update('B_C_08c_wasserstein_distance_classe_2', classe_2_w_distance)
            self.writer.add_histogram('B_C_09b_distr_classe_3_samples',torch.FloatTensor(utils.get_smiles_properties(utils.get_values_by_classe(samples_valid_smiles, samples_valid_classes, 2),self.data_loader.get_type_property())))
            classe_3_w_distance = metrics.wasserstein_distance_score(utils.get_smiles_properties(utils.get_values_by_classe(samples_valid_smiles, samples_valid_classes, 2),self.data_loader.get_type_property()), utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 2))
            self.train_metrics.update('B_C_09c_wasserstein_distance_classe_3', classe_3_w_distance)
            self.train_metrics.update('B_C_10_wasserstein_distance_sum_classes', classe_1_w_distance+classe_2_w_distance+classe_3_w_distance)
          except Exception:
            pass

          try:
            accuracy_match_classes, true_classes = metrics.accuracy_match_classes(samples_valid_mols, samples_valid_classes, self.data_loader.get_type_property(), self.data_loader.get_num_classes(), self.data_loader.get_classes_breakpoints())
            self.train_metrics.update('B_C_11_classes_match', accuracy_match_classes)
          except Exception:
            accuracy_match_classes = 0
          
          # image
          try:
            samples_valid_mols_cond = []
            max_mol_by_classe = 4
            classe_start = 0
            cpt_mol_by_class = 0
            # Select mols for each classe
            for idx, mol in enumerate(samples_valid_mols):
                if samples_valid_classes[idx] != classe_start:
                    classe_start = samples_valid_classes[idx]
                    cpt_mol_by_class = 0
                if cpt_mol_by_class <= max_mol_by_classe:
                    samples_valid_mols_cond.append(mol)
                    cpt_mol_by_class += 1

            self.writer.add_image('B_C_12_mol_samples', transforms.ToTensor()(Chem.Draw.MolsToGridImage(samples_valid_mols_cond, molsPerRow=max_mol_by_classe, subImgSize=(300,300))))
          except Exception:
            pass
          
          # sample
          # QM9DataLoaderSelfies
          try:
            if self.data_loader.__class__.__name__ == 'QM9DataLoaderSelfies':
              d = {'selfies':samples_valid_selfies,self.data_loader.get_type_property():prop_samples,'predicted classe':samples_valid_classes.cpu(),'true classe':true_classes.cpu()}
              df = pd.DataFrame(d)
              df.to_csv(f'{self.result_dir}/selfies-c-{step}.csv', index=False)
              
            d = {'smiles':samples_valid_smiles,self.data_loader.get_type_property():prop_samples,'predicted classe':samples_valid_classes.cpu(),'true classe':true_classes.cpu()}
            df = pd.DataFrame(d)
            df.to_csv(f'{self.result_dir}/smiles-c-{step}.csv', index=False)
          except Exception:
            pass
          
        else:
          accuracy_match_classes = 0

        # only first eval
        if self.step == self.eval_and_sample_every:
            self.writer.add_histogram('B_C_06a_distr_prop_train',torch.FloatTensor(self.data_loader.get_train_prop()))
            self.writer.add_histogram('B_C_07a_distr_classe_1_train',torch.FloatTensor(utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 0)))
            self.writer.add_histogram('B_C_08a_distr_classe_2_train',torch.FloatTensor(utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 1)))
            self.writer.add_histogram('B_C_09a_distr_classe_3_train',torch.FloatTensor(utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 2)))


        return accuracy_match_classes
