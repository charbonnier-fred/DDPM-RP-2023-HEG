import numpy as np
import os
import torch
import utils
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.DataStructs import FingerprintSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # suppress error messages
from guacamol.utils.sampling_helpers import sample_valid_molecules
from guacamol.distribution_learning_benchmark import ValidityBenchmark, UniquenessBenchmark, NoveltyBenchmark, \
    KLDivBenchmark
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from scipy.stats import wasserstein_distance
import sys
from typing import List
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer



class MockGenerator(DistributionMatchingGenerator):
    """
    Mock generator that returns pre-defined molecules,
    possibly split in several calls
    """

    def __init__(self, molecules: List[str]) -> None:
        self.molecules = molecules
        self.cursor = 0

    def generate(self, number_samples: int) -> List[str]:
        end = self.cursor + number_samples

        sampled_molecules = self.molecules[self.cursor:end]
        self.cursor = end
        return sampled_molecules

# Calculates the validity score with guacamol
def validity_score(smiles):
    with torch.no_grad():
      generator = MockGenerator(smiles)
      benchmark = ValidityBenchmark(number_samples=len(smiles))
    return benchmark.assess_model(generator).score

# Calculates the uniqueness score with guacamol
def uniqueness_score(smiles):
    with torch.no_grad():
      generator = MockGenerator(smiles)
      generator = MockGenerator(sample_valid_molecules(generator, len(smiles)))
      benchmark = UniquenessBenchmark(number_samples=len(smiles))
    return benchmark.assess_model(generator).score

# Calculates the novelty score with guacamol
def novelty_score(gen_smiles, train_smiles):
    with torch.no_grad():
      generator = MockGenerator(gen_smiles)
      generator = MockGenerator(sample_valid_molecules(generator, len(gen_smiles)))
      benchmark = NoveltyBenchmark(number_samples=len(gen_smiles), training_set=train_smiles)
    return benchmark.assess_model(generator).score

# Calculates the KL divergence score with guacamol
def KLdiv_score(gen_smiles, train_smiles):
    with torch.no_grad():
      generator = MockGenerator(gen_smiles)
      generator = MockGenerator(sample_valid_molecules(generator, len(gen_smiles)))
      benchmark = KLDivBenchmark(number_samples=len(gen_smiles), training_set=train_smiles)
      result = benchmark.assess_model(generator)
    return result.score

# Calculates syntax validity accuracy
def accuracy_syntax_validity(valid_samples, all_samples):
    with torch.no_grad():
      return len(valid_samples) / len(all_samples)

# Calculates classes match accuracy
def accuracy_match_classes(mols, classes, prop, num_classes, breakpoints):
  with torch.no_grad():
    if prop == "Weight":
        prop_values = [Chem.Descriptors.MolWt(mol) for mol in mols]
    elif prop == "LogP":
        prop_values = [Chem.Descriptors.MolLogP(mol) for mol in mols]
    elif prop == "QED":
        prop_values = [Chem.QED.default(mol) for mol in mols]
    elif prop == "SAS":
        prop_values = [sascorer.calculateScore(mol) for mol in mols]
    true_classes, _ = utils.discretize_continuous_values(prop_values, num_classes, breakpoints)
    nb_diff = 0
    for c1, c2 in zip(classes, true_classes):
        if c1 != c2:
            nb_diff += 1
  return (len(classes)-nb_diff)/len(classes), true_classes

# Calculate similarity
def tanimoto_similarity(database_mols, query_mol):
  """Compare generated molecules to database by Tanimoto similarity."""
  with torch.no_grad():
    # convert Mol to datastructure type
    fps = [FingerprintMol(m) for m in database_mols]
    
    # set a query molecule to compare against database
    query = FingerprintMol(query_mol)
    
    similarities = []
    
    # loop through to find Tanimoto similarity
    for idx, f in enumerate(fps):
        # tuple: (idx, similarity)
        similarities.append((idx, FingerprintSimilarity(query, f)))
    
    # sort sim using the similarities
    similarities.sort(key=lambda x:x[1], reverse=True)
  
  return similarities

# Calculates top 3 similarity score
def top3_tanimoto_similarity_score(gen_mols, train_mols):
    with torch.no_grad():
      scores = 0
      for gen_mol in gen_mols:
         tanimoto_scores = tanimoto_similarity(train_mols, gen_mol)
         scores_list = [tanimoto_score[1] for tanimoto_score in tanimoto_scores[:3]]
         scores += sum(scores_list)
    return scores / (len(gen_mols)*3)

# Calculates Wasserstein distance
def wasserstein_distance_score(gen_values, train_values):
  w_distance = wasserstein_distance(gen_values, train_values)
  return w_distance
