import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import rdinchi
import warnings
import csv

# Needed information at some point:
# - a standard name for each compound with left-padded indices (e.g., IBM000000001)
# - the split for each compound (train, val, test)
# - the SMILES for each compound
# - the InChI for each compound (I'm not sure if both SMILES and InChI are needed, but let's include both for now)
# - the formula for each compound
# - the mass of the precursor ion for each spectrum
# - the fragments for each spectrum
#       - the m/z values for each fragment
#       - the intensity values for each fragment
#       - the ion of each fragment (seems to be just the precursor ion [M+H]+ or [M+Na]+ repeated for each fragment)

# Required fields for each compound for DiffMS
REQUIRED_FIELDS = ['split', 'smiles', 'inchi', 'formula', 'mass',
                   'fragments', 'mz', 'intensity', 'ion']
# Fields that are isolated and cannot be extrapolated from other fields
ISOLATED_FIELDS = ['fragments', 'mz', 'intensity', 'ion'] # Exclude split so it can be created later if desired
# Fields that are structural representations of the compound
STRUCTURAL_FIELDS = ['smiles', 'inchi']

# Functions to handle the required fields for each compound and extrapolate data
#####################################################################################################################
def getRequiredFields():
    """
    Returns a dictionary containing the required data fields for each compound.
    Returns:
        dict: A dictionary containing the required data fields for each compound.
    """
    req_data = {field: None for field in REQUIRED_FIELDS}
    return req_data


def extrapolateCurrentData(req_data: dict):
    """
    Uses the information in the req_data dictionary to extrapolate the current data. Will raise an error
    if the required data is not present in the dictionary.
    Args:
        req_data (dict): A dictionary containing the required data for extrapolation. Unfilled fields are None.
    Returns:
        req_data (dict): Input req_dict modified to include extrapolated data.
    """
    empty_fields = [k for k, v in req_data.items() if v is None]
    if not empty_fields:
        return req_data  # Nothing to extrapolate, return as is

    # Check for fields that have no way to extrapolate
    cannot_extrapolate = set(ISOLATED_FIELDS) & set(empty_fields)
    if cannot_extrapolate:
        raise ValueError(f"Cannot extrapolate fields: {cannot_extrapolate}")

    # Check for structural representations
    req_to_extrapolate = set(STRUCTURAL_FIELDS) & set(empty_fields)
    if not req_to_extrapolate:
        raise ValueError("No fields to extrapolate from! Please provide at least one of the following: "
                         f"{req_to_extrapolate}")
    
    # If we have InChI, we can extrapolate the SMILES
    if 'smiles' in empty_fields:
        mol = Chem.MolFromInchi(req_data['inchi'])
        if mol is None:
            raise ValueError(f"Invalid InChI string: {req_data['inchi']}")
        req_data['smiles'] = Chem.MolToSmiles(mol, isomericSmiles=False)

    # At this point, we have the SMILES. Ensure that it is valid and canonicalize it and remove stereochemistry.
    mol = Chem.MolFromSmiles(req_data['smiles'])
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    req_data['smiles'] = Chem.CanonSmiles(smiles, useChiral=False)
    mol = Chem.MolFromSmiles(req_data['smiles'])
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {req_data['smiles']}")

    # SMILES can be used to extrapolate the InChI
    if 'inchi' in empty_fields:
        req_data['inchi'] = Chem.MolToInchi(mol)

    # SMILES can be used to extrapolate the formula
    if 'formula' in empty_fields:
        req_data['formula'] = Chem.rdMolDescriptors.CalcMolFormula(mol)

    # SMILES can be used to extrapolate the mass
    if 'mass' in empty_fields:
        req_data['mass'] = ExactMolWt(mol)

    return req_data


def extrapolateAllData(req_data: dict):
    """
    Uses the information in the req_data dictionary to extrapolate all data for each compound.
    """
    for cmpd, data in tqdm(req_data.items()):
        req_data[cmpd] = extrapolateCurrentData(data)
#####################################################################################################################
        

# Functions to write DiffMS files
#####################################################################################################################
def writeHelperFile(save_dir: str, save_name: str, req_data: dict):
    """
    Write files that contain the required data for DiffMS for each compound as a JSON file.
    Args:
        save_dir (str): The directory to save the files to.
        save_name (str): The name of the file to write the data to. It will be a .json file.
        req_data (dict): A dictionary containing the required data for each compound.
            Assumes the keys are left-padded indices (e.g., IBM000000001) and the values are dictionaries
            containing the required fields.
    """
    # Verify the data is in the correct format
    blank_reqs = [k for k in getRequiredFields().keys()]
    if np.any([k not in req_data for k in blank_reqs]):
        raise ValueError(f"Required fields {blank_reqs} not found in req_data! Please ensure the data is in the correct format.")
    # Ensure that the required fields are filled
    for k, v in req_data.items():
        if k == 'split':
            continue
        if v is None:
            raise ValueError(f"Required field '{k}' is not filled for compound {k}! Please ensure the data is in the correct format.")

    # Ensure save_name has the correct file extension
    if '.' not in save_name:
        save_name += '.json'
    if not save_name.endswith('.json'):
        base, _ = os.path.splitext(save_name)
        save_name = base + '.json'
    save_path = os.path.join(save_dir, save_name)

    # Write the helper file in the appropriate format
    with open(save_path, 'w') as f:
        json.dump(req_data, f)
        

def writeSplitFile(save_dir: str, read_dir: str, cmpd_name: str, split_file: str):
    """
    Writes the split files in the appropriate format for DiffMS.
    Args:
        save_dir (str): The directory to save the split files to.
        read_dir (str): The directory to read the compound information from. See writeHelperFile for the expected format.
        cmpd_name (str): The name of the compound in the standard format (e.g., IBM000000001).
        split_file (str): The path to the split file containing the train, val, and test splits.
    """
    split_dict = np.load(split_file, allow_pickle=True).item()
    # Ensure that the split_dict is in the correct format
    if not isinstance(split_dict, dict):
        raise ValueError(f"Split file {split_file} is not a dictionary!")
    # Distribute splits to their own list
    train_idxs = split_dict.get('train', list())
    val_idxs = split_dict.get('val', list())
    test_idxs = split_dict.get('test', list())
    
    # Ensure save_name has the correct file extension
    if '.' not in save_name:
        save_name += '.tsv'
    if not save_name.endswith('.tsv'):
        base, _ = os.path.splitext(save_name)
        save_name = base + '.tsv'
    save_path = os.path.join(save_dir, save_name)

    # Get all of the compound names from the read directory
    all_read_filenames = os.listdir(read_dir)
    all_names = sorted([f.replace('.json', '') for f in all_read_filenames if f.endswith('.json')])

    # Write the split file in the appropriate format
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Write the header
        header = ['name', 'split']
        writer.writerow(header)
        all_data = list()
        for i, name in tqdm(enumerate(all_names), desc='Writing split file'):
            if i in train_idxs:
                split = 'train'
            elif i in val_idxs:
                split = 'val'
            elif i in test_idxs:
                split = 'test'
            else:
                warnings.warn(f"Index {i} not found in any split!")
            all_data.append([name, split])
        # Write the data
        writer.writerows(all_data)


def writeLabelsFile(save_dir: str, read_dir: str, name: str):
    """
    Writes the labels file in the appropriate format for DiffMS.
    Args:
        save_dir (str): The directory to save the labels file to.
        read_dir (str): The directory to read the compound information from. See writeHelperFile for the expected format.
        name (str): The name of the file to write the labels to. It will be a .tsv file full of text.
    """


# NOTE: It appears DiffMS can use more than one energy based on the spectrum files.
#       Some files have multiple >ms2peaks sections, which suggests some difference between these spectra.
#       No difference is explicitly stated in the CANOPUS dataset, but the sections vary in length.
#       For now, use one energy to match our current implementation.
def writeSpectrumFile(save_dir: str, read_dir: str, name: str):
    """
    Writes the spectrum files in the appropriate format for DiffMS.
    Args:
        save_dir (str): The directory to save the spectrum files to.
        read_dir (str): The directory to read the compound information from. See writeHelperFile for the expected format.
        save_name (str): The name of the file to write the spectra to. It will be a .ms file full of text.
    """
    # Read the compound information from the file
    read_path = os.path.join(read_dir, name + '.json')
    with open(read_path, 'r') as f:
        req_data = json.load(f)

    # Ensure save_name has the correct file extension
    if '.' not in name:
        name += '.ms'
    if not name.endswith('.ms'):
        base, _ = os.path.splitext(name)
        name = base + '.ms'
    save_path = os.path.join(save_dir, name)

    # Write the spectrum files in the appropriate format
    with open(save_path, 'w') as f:
        f.write(f">formula {req_data['formula']}\n")  # Write the formula
        f.write(f">parentmass {req_data['mass']}\n")  # Write the mass of the precursor ion
        f.write(f">ionization {req_data['ion']}\n")  # Write the ionization 
        f.write(f">InChi {req_data['inchi']}\n")  # Write the InChI
        f.write(f">InChIKey {rdinchi.InchiToInchiKey(req_data['inchi'])}\n")  # Write the InChI Key
        f.write(f"#smiles {req_data['smiles']}\n")  # Write the SMILES
        f.write(f"#instrumentation N/A\n")  # Placeholder for instrumentation
        f.write(f"#_FILE {name.replace('.ms', '')}\n")  # Write the file name without the extension
        f.write(f"#spectrumid {name.replace('.ms', '')}\n")  # Write the spectrum ID
        f.write(f"#_FILE_PATH {save_path}\n")  # Write the file path
        f.write(f"InchI {req_data['inchi']}\n")  # Write the InChI again, oddly
        f.write(f"#source {save_path}\n")  # Write the source file path
        f.write("\n")  # Blank line to separate the header from the data
        f.write(">ms2peaks\n")  # Start of the MS/MS peaks section
        for mz, intensity in zip(req_data['mz'], req_data['intensity']):
            f.write(f"{mz} {intensity}\n")  # Write the m/z and intensity pairs
#####################################################################################################################
            
# IBM data extraction specific functions
#####################################################################################################################
def extractIbmChunkData(chunk: pd.DataFrame, ion_mode: str, energy: int):
    """
    Takes in an open pandas DataFrame taht is a chunk of the IBM dataset and extracts the relevant data.
    Creates a generator that yields the extracted data for each compound in the chunk.
    Args:
        chunk (pandas.DataFrame): The chunk of the IBM dataset to extract data from.
        ion_mode (str): The ion mode for the MS/MS data ('positive' or 'negative').
        energy (int): The collision energy for the MS/MS data.
    Yields:
        req_data (dict): Input req_dict modified to include extracted data.
    """
    gathered_data = list()

    num_chunk_cmpds = len(chunk)
    for cmpd_i in tqdm(range(num_chunk_cmpds), desc='Extracting data from chunk'):
        curr_cmpd = chunk.iloc[cmpd_i]
        
        # Extract the SMILES for the compound
        smiles = curr_cmpd['smiles']

        # Extract the formula for the compound
        formula = curr_cmpd['molecular_formula']

        # Extract the MS/MS data for the compound
        ms_data = curr_cmpd[f'msms_{ion_mode}_{energy}ev']

        # Extract the fragment information for the compound
        fragment_data = curr_cmpd[f'msms_fragments_{ion_mode}']

        # IBM paper assumes all compounds are [M+H]+ or [M-H]- ions, so we can use the mass of the precursor ion
        ion = '[M+H]+' if ion_mode == 'positive' else '[M-H]-'

        # Parse the data into a desired format
        # NOTE: This could get moved to an external function if information is typically extracted in this way,
        #   but for now, we'll keep it here for simplicity.
        # Convert the (m/z, intensity) pairs into two separate lists for each spectrum
        mz_values, intensity_values = list(), list()
        for mz_intensity_pair in ms_data:
            mz_values.append(float(mz_intensity_pair[0])) # m/z value, ensure it's a float
            intensity_values.append(float(mz_intensity_pair[1])) # intensity value, ensure it's a float
        mz_values, intensity_values = np.array(mz_values), np.array(intensity_values)

        # Fragment info is stored as (m/z, fragment) pairs for every fragment in all spectra for a particular ion mode.
        #   Get the intensity of each relevant fragment s.t. the fragment info can be parallel to the m/z and intensity values.
        relevant_fragments = np.array([''] * len(mz_values), dtype=object)  # Initialize an array for relevant fragments
        num_found_fragments = 0
        for mz_frag_pair in fragment_data:
            frag_mz = float(mz_frag_pair[0])
            mz_idx = np.where(mz_values == frag_mz)[0]
            if len(mz_idx):
                relevant_fragments[mz_idx[0]] = mz_frag_pair[1]
                num_found_fragments += 1
        if num_found_fragments != len(mz_values):
            warnings.warn(f"Not all fragments were found in the m/z values for compound {cmpd_i}! Skipping compound")
            continue  # Skip this compound if not all fragments were found

        # Create the required data dictionary for the current compound
        req_data = getRequiredFields()
        # DiffMS sorts by intensity, so we will do the same
        sort_idxs = np.argsort(intensity_values)
        req_data['mz'] = mz_values[sort_idxs]
        req_data['intensity'] = intensity_values[sort_idxs]
        req_data['fragments'] = relevant_fragments[sort_idxs]
        req_data['formula'] = formula
        req_data['smiles'] = smiles
        req_data['ion'] = ion
        req_data = extrapolateCurrentData(req_data)  # Extrapolate the data for the current compound
        gathered_data.append(req_data)

    yield from gathered_data  # Yield the extracted data for the each compound


def extractIbmData(cfg):
    """
    Extracts the IBM dataset from the specified directory and returns the relevant data.
    Assumes the only files within the directory are the IBM dataset files.
    Args:
        cfg (dict): Configuration dictionary containing parameters like ion_mode and energy.
    Returns:
        stored_data: A dictionary containing the extracted data for each compound.
    """
    import pyarrow.parquet as pq

    data_dir = cfg.data_dir

    # Get all of the data files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if not all_files:
        raise ValueError(f"No .parquet files found in directory: {data_dir}")
    
    # Get the total number of compounds in the dataset
    num_total_cmpds = 0
    for file in tqdm(all_files, desc='Counting total compounds in dataset'):
        data = pq.ParquetFile(file)
        num_total_cmpds += data.metadata.num_rows
    print(f"Total number of compounds in dataset: {num_total_cmpds}")

    # Create a dictionary to store the data for each compound
    num_chars = len(str(num_total_cmpds))  # Number of characters in the total number of compounds
    stored_data = {str(i).zfill(num_chars): getRequiredFields() for i in range(num_total_cmpds)}

    # Iterate through each file and extract the data using extractIbmChunkData generator
    all_keys = list(stored_data.keys())  # List of keys for the stored data
    for file in tqdm(all_files, desc='Extracting data from files'):
        data = pd.read_parquet(file, engine='pyarrow')
        data_generator = extractIbmChunkData(stored_data, data, ion_mode=cfg.ion_mode, energy=cfg.energy)
        for cmpd_i, cmpd_data in enumerate(data_generator):
            cur_key = all_keys[cmpd_i]
            stored_data[cur_key] = cmpd_data
#####################################################################################################################