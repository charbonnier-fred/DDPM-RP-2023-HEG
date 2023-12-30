import deepchem as dc
import deepsmiles
from guacamol.utils.chemistry import canonicalize
import numpy as np
import os
import pandas as pd
import selfies as sf
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Helper
def keys_int(symbol_to_int):
  d={}
  i=0
  for key in symbol_to_int.keys():
    d[i]=key
    i+=1
  return d

# Preprocess SMILES (SMILES -> DeepSMILES, replace multi-caracters alphabet and extract alphabet)
def preprocess_smiles(smiles, deepsmiles_converter):
  smiles = smiles.apply(deepsmiles_converter.encode)
  replace_dict = {'Cn': 'a','[C@@H]':'b', '[C@@]':'d', '[C@H+]':'e', '[C@H]':'f', '[C@]':'g', '[CH+]':'h', '[CH-]':'i', '[CH2+]':'j', '[CH2-]':'k', '[CH]':'l', '[H]':'m', '[N+]':'p', '[N@@H+]':'q', '[N@H+]':'r', '[NH+]':'s', '[NH-]':'t', '[NH2+]':'u', '[NH3+]':'v', '[O-]':'w', '[OH+]':'x', '[cH+]':'y', '[cH-]':'z', '[n+]':'A', '[nH+]':'B', '[nH]':'D', '\\':'E'}
  smiles_alphabet = ['#', ')', '-', '.', '/', '3', '4', '5', '6', '7', '8', '9', '=', 'A','B','C', 'D','E', 'F', 'N', 'O', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s','t','u','v','w','x','y','z']
  
  for pattern, repl in replace_dict.items():
    smiles = smiles.str.replace(
        pattern, repl
    )
  largest_smiles_len = len(max(smiles, key=len))
  return smiles, replace_dict, smiles_alphabet, largest_smiles_len

# Convert DeepSMILES to SMILES
def decode_deepsmiles(smiles, deepsmiles_converter):
  try:
    smiles = deepsmiles_converter.decode(smiles)
    return smiles
  except deepsmiles.DecodeError as e:
    return smiles

# Postprocess SMILES (replace multi-caracters alphabet, DeepSMILES -> SMILES)
def postprocess_smiles(smiles, replace_dict, deepsmiles_converter):
  for pattern, repl in replace_dict.items():
    smiles = smiles.str.replace(
        repl, pattern
    )
  smiles = smiles.apply(decode_deepsmiles,deepsmiles_converter=deepsmiles_converter)
  return smiles

# Convert SMILES to SELFIES
def smiles_to_selfies(smiles):
   selfies_list = np.asanyarray(smiles.apply(sf.encoder))
   return selfies_list

# Calculates SELFIES alphabet
def get_selfies_features(selfies, canonicalize_smiles):
  selfies_alphabet = sf.get_alphabet_from_selfies(selfies)
  if '[/F]' not in selfies_alphabet:
    selfies_alphabet.add('[/F]')
  if '[\\CH1-1]' not in selfies_alphabet:
    selfies_alphabet.add('[\\CH1-1]')
  # Padding
  if canonicalize_smiles == False:
    selfies_alphabet.add('[/C@@H1]')
    selfies_alphabet.add('[/C@H1]')
    selfies_alphabet.add('[/CH1+1]')
    selfies_alphabet.add('[/CH1-1]')

  selfies_alphabet.add('[nop]')  # Add the "no operation" symbol as a padding character
  selfies_alphabet.add('.') 
  selfies_alphabet = list(sorted(selfies_alphabet))
  largest_selfie_len = max(sf.len_selfies(s) for s in selfies)
  symbol_to_int = dict((c, i) for i, c in enumerate(selfies_alphabet))
  int_mol=keys_int(symbol_to_int)
  return largest_selfie_len, selfies_alphabet, symbol_to_int, int_mol

# Convert SELFIES to continous matrix
def selfies_to_continous_mols(selfies, largest_selfie_len, selfies_alphabet, symbol_to_int):
   onehots = torch.zeros(len(selfies), largest_selfie_len, len(selfies_alphabet), dtype=torch.float)
   for i, selfie in enumerate(selfies):
    one_hot = sf.selfies_to_encoding(selfie, symbol_to_int, largest_selfie_len, enc_type='one_hot')
    onehots[i, :, :] = torch.tensor(one_hot, dtype=torch.float32)
   dequantized_onehots = onehots.add(torch.rand(onehots.shape, dtype=torch.float32))
   continous_mols = dequantized_onehots.div(2)
   return continous_mols

# Convert SMILES to continous matrix
def smiles_to_continous_mols(smiles, featurizer):
   encodings = featurizer.featurize(smiles)
   onehots = torch.tensor(encodings, dtype=torch.float32)
   dequantized_onehots = onehots.add(torch.rand(onehots.shape, dtype=torch.float32))
   continous_mols = dequantized_onehots.div(2)
   return continous_mols

# Convert one SELFIES to continous matrix
def one_selfies_to_continous_mol(selfies, largest_selfie_len, symbol_to_int):
   one_hot = sf.selfies_to_encoding(selfies, symbol_to_int, largest_selfie_len, enc_type='one_hot')
   onehot = torch.tensor(one_hot, dtype=torch.float32)
   dequantized_onehot = onehot.add(torch.rand(onehot.shape, dtype=torch.float32))
   continous_mol = dequantized_onehot.div(2)
   return continous_mol

# Convert one SMILES to continous matrix
def one_smiles_to_continous_mol(smiles, featurizer):
  encodings = featurizer.featurize([smiles])
  onehot = torch.tensor(encodings.squeeze(), dtype=torch.float32)
  dequantized_onehot = onehot.add(torch.rand(onehot.shape, dtype=torch.float32))
  continous_mol = dequantized_onehot.div(2)
  return continous_mol

# Convert continous matrix to SELFIES
def continous_mols_to_selfies(continous_mols, selfies_alphabet, int_mol):
   denormalized_data = continous_mols * 2
   quantized_data = torch.floor(denormalized_data)
   quantized_data = torch.clip(quantized_data, 0, 1)
   for mol in quantized_data:
    for letter in mol:
      if all(elem == 0 for elem in letter):
        letter[len(selfies_alphabet)-1] = 1
   selfies = [sf.encoding_to_selfies(mol.cpu().tolist(), int_mol, enc_type="one_hot") for mol in quantized_data]
   return selfies

# Convert continous matrix to SMILES
def continous_mols_to_smiles(continous_mols, featurizer, replace_dict, deepsmiles_converter):
  denormalized_data = continous_mols * 2
  quantized_data = torch.floor(denormalized_data)
  quantized_data = torch.clip(quantized_data, 0, 1)
  for mol in quantized_data:
    for letter in mol:
      if all(elem == 0 for elem in letter):
        letter[-1] = 1
  smiles = [featurizer.untransform(mol.cpu().tolist()) for mol in quantized_data]
  df_smiles = pd.DataFrame(smiles, columns=['smiles'])
  smiles = postprocess_smiles(df_smiles['smiles'], replace_dict, deepsmiles_converter)
  return smiles.values.tolist()

# Return valid SELFIES
def get_valid_selfies(selfies):
  valid_selfies = []
  for _, selfie in enumerate(selfies):
    try:
      if Chem.MolFromSmiles(sf.decoder(selfie), sanitize=True) is not None:
         valid_selfies.append(selfie)
    except Exception:
      pass
  return valid_selfies

# Return valid SMILES
def get_valid_smiles(smiles):
  valid_smiles = []
  for _, smile in enumerate(smiles):
    try:
      if Chem.MolFromSmiles(smile, sanitize=True) is not None:
         valid_smiles.append(smile)
    except Exception:
      pass
  return valid_smiles

# Return valid SMILES and its class
def get_valid_smiles_and_classes(smiles_to_split, classes):
  valid_smiles, valid_classes = [], []
  for idx, smiles in enumerate(smiles_to_split):
    try:
      if Chem.MolFromSmiles(smiles, sanitize=True) is not None:
          valid_smiles.append(smiles)
          valid_classes.append(classes[idx])
    except Exception:
      pass
  return valid_smiles, torch.tensor(valid_classes)

# Return valid SELFIES and its class
def get_valid_selfies_and_classes(selfies_to_split, classes):
  valid_selfies, valid_classes = [], []
  for idx, selfies in enumerate(selfies_to_split):
    try:
      if Chem.MolFromSmiles(sf.decoder(selfies), sanitize=True) is not None:
          valid_selfies.append(selfies)
          valid_classes.append(classes[idx])
    except Exception:
      pass
  return valid_selfies, torch.tensor(valid_classes)

# Convert SELFIES to Mol
def selfies_to_mols(selfies_to_convert):
  valid_count = 0
  valid_selfies, invalid_selfies = [], []
  mols = []
  for idx, selfies in enumerate(selfies_to_convert):
    try:
      if Chem.MolFromSmiles(sf.decoder(selfies), sanitize=True) is not None:
          valid_count += 1
          valid_selfies.append(selfies)
          mols.append(Chem.MolFromSmiles(sf.decoder(selfies)))
      else:
        invalid_selfies.append(selfies)
    except Exception:
      pass
  return mols, valid_selfies, valid_count

# Convert SMILES to Mol
def smiles_to_mols(smiles_to_convert):
  valid_count = 0
  valid_smiles, invalid_smiles = [], []
  mols = []
  for idx, smiles in enumerate(smiles_to_convert):
    try:
      if Chem.MolFromSmiles(smiles, sanitize=True) is not None:
          valid_count += 1
          valid_smiles.append(smiles)
          mols.append(Chem.MolFromSmiles(smiles))
      else:
        invalid_smiles.append(smiles)
    except Exception:
      pass
  return mols, valid_smiles, valid_count

# Convert Mol to SMILES
def mols_to_smiles(mols):
   smiles = [Chem.MolToSmiles(mol) for mol in mols]
   return smiles

# Continous matrix to int values
def discretize_continuous_values(values, num_classes, breakpoints = None):
  if breakpoints is None:
    sorted_values = sorted(values)
    breakpoints = [sorted_values[i * len(values) // num_classes] for i in range(1, num_classes)]
  discretized_values = []
  for value in values:
      class_value = sum(value > breakpoint for breakpoint in breakpoints)
      discretized_values.append(class_value)
  return torch.tensor(discretized_values), breakpoints

# Canonicalize SMILES
def canonicalize_smiles(smiles):
  smiles_canonicalized = [canonicalize(smile) for smile in smiles]
  return smiles_canonicalized

# Canonicalize one SMILES
def canonicalize_one_smiles(smiles):
  return canonicalize(smiles)

# Return Mol properties
def get_mols_properties(mols, prop):
  if prop == "Weight":
    molsWt = [Chem.Descriptors.MolWt(mol) for mol in mols]
    return molsWt
  elif prop == "LogP":
    molsLogP = [Chem.Descriptors.MolLogP(mol) for mol in mols]
    return molsLogP
  elif prop == "QED":
    molsQED = [Chem.QED.default(mol) for mol in mols]
    return molsQED
  elif prop == "SAS":
    molsSAS = [sascorer.calculateScore(mol) for mol in mols]
    return molsSAS

# Return SELFIES properties
def get_selfies_properties(selfies, prop):
  props = []
  for _, selfie in enumerate(selfies):
    try:
      mol = Chem.MolFromSmiles(sf.decoder(selfie), sanitize=True)
      if mol is not None:
         if prop == "Weight":
            props.append(Chem.Descriptors.MolWt(mol))
         elif prop == "LogP":
            props.append(Chem.Descriptors.MolLogP(mol))
         elif prop == "QED":
            props.append(Chem.QED.default(mol))
         elif prop == "SAS":
            props.append(sascorer.calculateScore(mol))
    except Exception:
      pass
  return props

# Return SMILES properties
def get_smiles_properties(smiles, prop):
  props = []
  for _, smile in enumerate(smiles):
    try:
      mol = Chem.MolFromSmiles(smile, sanitize=True)
      if mol is not None:
         if prop == "Weight":
            props.append(Chem.Descriptors.MolWt(mol))
         elif prop == "LogP":
            props.append(Chem.Descriptors.MolLogP(mol))
         elif prop == "QED":
            props.append(Chem.QED.default(mol))
         elif prop == "SAS":
            props.append(sascorer.calculateScore(mol))
    except Exception:
      pass
  return props
