import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import rdinchi
import warnings
import csv
import hydra
from typing import Generator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings for cleaner output

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
REQUIRED_FIELDS = ['split', 'smiles', 'inchi', 'inchikey', 'formula', 'mass',
                   'fragments', 'mz', 'intensity', 'ion']
# Fields that are isolated and cannot be extrapolated from other fields
ISOLATED_FIELDS = ['fragments', 'mz', 'intensity', 'ion'] # Exclude split so it can be created later if desired
# Fields that are structural representations of the compound
STRUCTURAL_FIELDS = ['smiles', 'inchi']

PROTON_MASS = 1.0072764665789 # Da

# Class to write DiffMS files
#####################################################################################################################
class DiffMSFileWriter:
    """
    A class to write DiffMS files for compounds in the appropriate format.
    """

    def __init__(self,
                 dataset_name: str,
                 total_cmpds: int,
                 save_dir: str,
                 remove_hydrogen: bool = False):
        """
        Initialize the DiffMSFileWriter.
        Args:
            dataset_name (str): The name of the dataset. Will be used to save files in a standard format (e.g., IBM000000001).
            total_cmpds (int): The total number of compounds in the entire dataset.
            save_dir (str): The directory to save the DiffMS files to.
            remove_hydrogen (bool), optional: Whether to remove an extra hydrogen from each fragment to account for adduct ion. Default is False.
        """
        # Make sure read and save directories are provided
        assert save_dir is not None, "Please provide a save directory for the dataset!"

        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Ensure the required subdirectories exist
        spectrum_dir = os.path.join(save_dir, 'spec_files')
        subformula_dir = os.path.join(save_dir, 'subformulae')
        helper_dir = os.path.join(save_dir, 'helper_files')
        check_dirs = [spectrum_dir, subformula_dir, helper_dir]
        for d in check_dirs:
            if not os.path.exists(d):
                os.makedirs(d)

        # Store variables as class attributes
        self.save_dir = save_dir
        self.spectrum_dir = spectrum_dir
        self.subformula_dir = subformula_dir
        self.helper_dir = helper_dir
        self.remove_hydrogen = remove_hydrogen
        self.dataset_name = dataset_name
        self.total_cmpds = total_cmpds
        self.cmpd_counter = 0


    def _nameCompound(self):
        """
        Generate a standard name for the compound with left-padding.
        """
        cmpd_name = self.dataset_name + str(self.cmpd_counter).zfill(len(str(self.total_cmpds)))  # Left-pad the name with zeros
        self.cur_name = cmpd_name
        self.cmpd_counter += 1
        return cmpd_name

    
    def _getCurrentName(self):
        """
        Get the current name of the compound.
        """
        return self.cur_name


    def _writeHelperFile(self, req_data: dict):
        """
        Write files that contain the required data for DiffMS for each compound as a JSON file.
        Args:
            req_data (dict): A dictionary containing the required data for the compound.
        """
        # Verify the data is in the correct format
        if np.any([k not in req_data for k in REQUIRED_FIELDS]):
            raise ValueError(f"Required fields not found in req_data! Please ensure the data is in the correct format.")
        
        # Ensure that the required fields are filled
        for k, v in req_data.items():
            if k == 'split':
                continue
            if v is None:
                raise ValueError(f"Required field '{k}' is not filled for compound! Please ensure the data is in the correct format.")

        # Name the file with the compound name in standard format
        cmpd_name = self._nameCompound()

        # Ensure save_name has the correct file extension
        save_path = validateFileExtension(cmpd_name, '.json', self.helper_dir)

        # Change any np arrays to lists for JSON serialization
        for k, v in req_data.items():
            if isinstance(v, np.ndarray):
                req_data[k] = v.tolist()

        # Write the helper file in the appropriate format
        with open(save_path, 'w') as f:
            json.dump(req_data, f)
            

    def _writeDiffMsSplitsFile(self, split_file: str, save_name: str = 'splits.tsv'):
        """
        Writes the split files in the appropriate format for DiffMS.
        Args:
            split_file (str): The path to the split file containing the train, val, and test splits. Default is 'splits.tsv'.
            save_name (str), optional: The name of the file to write the splits to. It will be a .tsv file.
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
        save_path = validateFileExtension(save_name, '.tsv', self.save_dir)

        # Get all of the compound names from the read directory
        all_read_filenames = os.listdir(self.helper_dir)
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


    def _writeDiffMsLabelsFile(self, save_name: str = 'labels.tsv'):
        """
        Writes the labels file in the appropriate format for DiffMS.
        Args:
            save_name (str), optional: The name of the file to write the labels to. It will be a .tsv file. Default is 'labels.tsv'.
        """
        # Get all of the compound names and info from the read directory
        all_read_filenames = sorted([f for f in os.listdir(self.helper_dir) if f.endswith('.json')])

        with open(os.path.join(self.save_dir, save_name), 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            # Write the header
            header = ['dataset', 'spec', 'name', 'ionization', 'formula', 'smiles', 'inchikey', 'instrument']
            writer.writerow(header)
            label_data_keys = ['ion', 'formula', 'smiles', 'inchikey']
            for filename in tqdm(all_read_filenames, desc='Writing labels file'):
                data_dict = json.load(open(os.path.join(self.helper_dir, filename), 'r'))
                # Data is expected to be in a certain format
                # Get the data from the file and put it in the correct order
                file_data = [data_dict.get(key, '') for key in label_data_keys]
                cmpd_name = filename.replace('.json', '')
                row_data = [self.dataset_name, cmpd_name, ''] + file_data + ['Unknown']  # '' for name and 'Unknown' for instrumentation as it's not provided
                writer.writerow(row_data)  # Write the row to the file


    # NOTE: It appears DiffMS can use more than one energy based on the spectrum files.
    #       Some files have multiple >ms2peaks sections, which suggests some difference between these spectra.
    #       No difference is explicitly stated in the CANOPUS dataset, but the sections vary in length.
    #       For now, use one energy to match our current implementation.
    def _writeDiffMsSpectrumFile(self, req_data: dict):
        """
        Writes the spectrum files in the appropriate format for DiffMS.
        Args:
            req_data (dict): A dictionary containing the required data for the compound. If None, will read from the helper_dir.
        """
        # Ensure save_name has the correct file extension
        save_path = validateFileExtension(self._getCurrentName(), '.ms', self.spectrum_dir)
        name = self._getCurrentName()
        
        # Write the spectrum files in the appropriate format
        with open(save_path, 'w') as f:
            f.write(f">compound {name}\n")  # Write the compound name
            f.write(f">formula {req_data['formula']}\n")  # Write the formula
            f.write(f">parentmass {req_data['mass']}\n")  # Write the mass of the precursor ion
            f.write(f">ionization {req_data['ion']}\n")  # Write the ionization 
            f.write(f">InChi {req_data['inchi']}\n")  # Write the InChI
            f.write(f">InChIKey {req_data['inchikey']}\n")  # Write the InChI Key
            f.write(f"#smiles {req_data['smiles']}\n")  # Write the SMILES
            f.write(f"#instrumentation N/A\n")  # Placeholder for instrumentation
            f.write(f"#_FILE {name}\n")  # Write the file name without the extension
            f.write(f"#spectrumid {name}\n")  # Write the spectrum ID
            f.write(f"#_FILE_PATH {save_path}\n")  # Write the file path
            f.write(f"#InchI {req_data['inchi']}\n")  # Write the InChI again, oddly
            f.write(f"#source {save_path}\n")  # Write the source file path
            f.write("\n")  # Blank line to separate the header from the data
            f.write(">ms2peaks\n")  # Start of the MS/MS peaks section
            for mz, intensity in zip(req_data['mz'], req_data['intensity']):
                f.write(f"{mz} {intensity}\n")  # Write the m/z and intensity pairs


    def _writeDiffMsSubformulaFile(self, req_data: dict):
        """
        Writes the subformula files in the appropriate format for DiffMS.
        Args:
            req_data (dict), optional: A dictionary containing the required data for the compound. If None, will read from the helper_dir.
        """
        # Ensure save_name has the correct file extension
        save_path = validateFileExtension(self._getCurrentName(), '.json', self.subformula_dir)

        # Get the fragments and remove adduct hydrogen if appropriate
        fragments = list(req_data['fragments'])
        if self.remove_hydrogen:
            fragments = [removeHydrogen(frag) for frag in fragments]

        # Get the information for the subformula file
        output_table = {
            'mz': list(req_data['mz']),  # The m/z values for the fragments
            'ms2_inten': list(req_data['intensity']),  # The intensity values for the fragments
            'mono_mass': list(), # Not sure what this is supposed to be
            'abs_mass_diff': list(), # Not sure what this is supposed to be
            'mass_diff': list(),  # Not sure what this is supposed to be
            'formula': fragments,  # The fragments for the compound
            'ions': [req_data['ion']] * len(req_data['mz'])  # The ionization of the compound, repeated for each fragment
        }
        subformula_data = {
            'cand_form': req_data['formula'],  # The formula of the compound
            'cand_ion': req_data['ion'],  # The ionization of the compound
            'output_tbl': output_table,  # The output table
        }

        # Write the subformula data to a file
        json.dump(subformula_data, open(save_path, 'w'))


    def writeCompoundDiffMsFiles(self, req_data: dict = None):
        """
        Writes the DiffMS files for a single compound in the appropriate format.
        Args:
            spectrum_dir (str): The directory to save the spectrum files to.
            subformula_dir (str): The directory to save the subformula files to.
            helper_dir (str): The directory to read the compound information from. See writeHelperFile for the expected format.
            name (str): The name of the compound in standard format (e.g., IBM000000001) with left-padding.
            req_data (dict), optional: A dictionary containing the required data for the compound. If None, will read from the helper_dir.
        """
        if req_data is None:
            # Read the compound information from the file
            read_path = os.path.join(self.helper_dir, self._getCurrentName() + '.json')
            with open(read_path, 'r') as f:
                req_data = json.load(f)

        # Write the spectrum file
        self._writeDiffMsSpectrumFile(req_data)
        # Write the subformula file
        self._writeDiffMsSubformulaFile(req_data)


    def createCompundDiffMsFiles(self, cmpd_data: dict):
        """
        Write the DiffMS files that require only a single compound.
        Args:
            total_cmpds (int): The total number of compounds in the entire dataset.
            data_name (str): The name of the compound in standard format (e.g., IBM000000001) with left-padding.
            cmpd_data (dict): A dictionary containing the required data for the compound.
            helper_dir (str): The directory to read the compound information from. See writeHelperFile for the expected format.
            spec_dir (str): The directory to save the spectrum files to.
            subformula_dir (str): The directory to save the subformula files to.
        """
        self._writeHelperFile(cmpd_data)
        self.writeCompoundDiffMsFiles()

    def createDatasetDiffMsFiles(self, split_file, split_name, labels_name):
        """
        Writes the DiffMS files that require the entire dataset to be processed.
        Args:
            cfg (dict): Configuration dictionary containing parameters like save_dir and read_dir.
        """
        # Check to see if splits were provided. If not, create them
        # If path doesn't exist, assume comma-delimited string of percentages for train, val, test
        if not os.path.exists(split_file):
            train_percent, val_percent, test_percent = map(float, split_file.split(','))

            # Create the splits file if it doesn't exist
            createSplitsFile(self.save_dir, self.total_cmpds, train_percent=train_percent, val_percent=val_percent, test_percent=test_percent)
            split_file = os.path.join(self.save_dir, 'created_splits.npy')  # Update the split file path

        self._writeDiffMsSplitsFile(split_file, save_name=split_name)
        self._writeDiffMsLabelsFile(save_name=labels_name)

    def cleanupHelperFiles(self):
        """
        Removes the helper files directory to save space.
        """
        if os.path.exists(self.helper_dir):
            import shutil
            shutil.rmtree(self.helper_dir)
#####################################################################################################################
# END Class to write DiffMS files

# START Helper functions
#####################################################################################################################
def validateFileExtension(name: str, desired_extension: str, save_dir: str = None):
    """
    Ensures that the given file name has the desired extension.
    Args:
        name (str): The string to validate. Assumes periods are only used for file extensions.
        desired_extension (str): The desired file extension (e.g., '.json', '.tsv').
        save_dir (str), optional: The directory to save the file to. If provided, the full path will be returned.
    Returns:
        str: The validated file name with the desired extension.
    """
    # Make sure the extension has a period at the start
    if not desired_extension.startswith('.'):
        desired_extension = '.' + desired_extension

    # Validate that the name has the desired extension
    if '.' not in name:
        name += desired_extension
    if not name.endswith(desired_extension):
        base, _ = os.path.splitext(name)
        name = base + desired_extension
    if save_dir:
        return os.path.join(save_dir, name)
    return name


def createSplitsFile(save_dir: str, num_cmpds: int, train_percent: float = 80.0, val_percent: float = 10.0, test_percent: float = 10.0):
    """
    Creates a .npy splits file with the specified percentages for train, val, and test splits.
    """
    assert train_percent + val_percent + test_percent == 100, "Train, val, and test percentages must sum to 100!"

    # Create the split file with the specified percentages
    train_size = int(num_cmpds * (train_percent / 100))
    val_size = int(num_cmpds * (val_percent / 100))

    all_idxs = np.arange(num_cmpds)
    np.random.shuffle(all_idxs)  # Shuffle the indices to create random splits
    train_idxs = all_idxs[:train_size]
    val_idxs = all_idxs[train_size:train_size + val_size]
    test_idxs = all_idxs[train_size + val_size:]

    splits = {
        'train': train_idxs,
        'val': val_idxs,
        'test': test_idxs
    }
    
    # Save the splits to a .npy file
    save_path = os.path.join(save_dir, 'created_splits.npy')
    np.save(save_path, splits)


# The IBM dataset includes an extra hydrogen on each fragment to account for adduct ion. Remove that hydrogen
def removeHydrogen(formula):
    if "H" not in formula:
        return formula
    new_formula = formula.split("H")
    num_hs = ''
    # For each character after the H, check if it's a digit
    second_half = new_formula[1]
    if len(second_half) == 0: # If there is no second half, just return the first half
        return new_formula[0]
    elif not second_half[0].isdigit(): # If second half has no digit, remove the only hydrogen
        return new_formula[0] + new_formula[1]
    elif second_half.isdigit(): # If the second half is all digits, we can skip the loop and just subtract one
        num_hs = str(int(second_half) - 1)
        return new_formula[0] + "H" + num_hs
    done_digits = False
    i = 0
    while not done_digits:
        if not second_half[i].isdigit():
            done_digits = True
        else:
            num_hs += second_half[i]
            i += 1
    num_hs = second_half if second_half.isdigit() else num_hs
    new_formula[1] = second_half[i:]
    num_hs = str(int(num_hs) - 1)
    new_formula = new_formula[0] + "H" + num_hs + new_formula[1]
    return new_formula
#####################################################################################################################
# END Helper functions

# START Functions to handle the required fields for each compound and extrapolate data
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

    # At this point we have InChI and SMILES, so we can get the InChI Key
    if 'inchikey' in empty_fields:
        req_data['inchikey'] = rdinchi.InchiToInchiKey(req_data['inchi'])

    # SMILES can be used to extrapolate the formula
    if 'formula' in empty_fields:
        req_data['formula'] = CalcMolFormula(mol)

    # SMILES can be used to extrapolate the mass
    if 'mass' in empty_fields:
        req_data['mass'] = ExactMolWt(mol) + PROTON_MASS  # Assuming protonation for [M+H]+

    return req_data


def extrapolateAllData(req_data: dict):
    """
    Uses the information in the req_data dictionary to extrapolate all data for each compound.
    """
    for cmpd, data in tqdm(req_data.items()):
        req_data[cmpd] = extrapolateCurrentData(data)
#####################################################################################################################
# END Functions to handle the required fields for each compound and extrapolate data

# START IBM data extraction specific functions
#####################################################################################################################
# TODO: Make a generator that reads one-by-one instead of loading the whole list into memory
# def extractIbmChunkData(chunk: pd.DataFrame, ion_mode: str, energy: int):
#     """
#     Takes in an open pandas DataFrame taht is a chunk of the IBM dataset and extracts the relevant data.
#     Creates a generator that yields the extracted data for each compound in the chunk.
#     Args:
#         chunk (pandas.DataFrame): The chunk of the IBM dataset to extract data from.
#         ion_mode (str): The ion mode for the MS/MS data ('positive' or 'negative').
#         energy (int): The collision energy for the MS/MS data.
#     Yields:
#         req_data (dict): Input req_dict modified to include extracted data.
#     """
#     gathered_data = list()

#     num_chunk_cmpds = len(chunk)
#     for cmpd_i in tqdm(range(num_chunk_cmpds), desc='Extracting data from chunk'):
#         curr_cmpd = chunk.iloc[cmpd_i]
        
#         # Extract the SMILES for the compound
#         smiles = curr_cmpd['smiles']

#         # Extract the formula for the compound
#         formula = curr_cmpd['molecular_formula']

#         # Extract the MS/MS data for the compound
#         ms_data = curr_cmpd[f'msms_{ion_mode}_{energy}ev']

#         # Extract the fragment information for the compound
#         fragment_data = curr_cmpd[f'msms_fragments_{ion_mode}']

#         # IBM paper assumes all compounds are [M+H]+ or [M-H]- ions, so we can use the mass of the precursor ion
#         ion = '[M+H]+' if ion_mode == 'positive' else '[M-H]-'

#         # Parse the data into a desired format
#         # NOTE: This could get moved to an external function if information is typically extracted in this way,
#         #   but for now, we'll keep it here for simplicity.
#         # Convert the (m/z, intensity) pairs into two separate lists for each spectrum
#         mz_values, intensity_values = list(), list()
#         for mz_intensity_pair in ms_data:
#             mz_values.append(float(mz_intensity_pair[0])) # m/z value, ensure it's a float
#             intensity_values.append(float(mz_intensity_pair[1])) # intensity value, ensure it's a float
#         mz_values, intensity_values = np.array(mz_values), np.array(intensity_values)

#         # Fragment info is stored as (m/z, fragment) pairs for every fragment in all spectra for a particular ion mode.
#         #   Get the intensity of each relevant fragment s.t. the fragment info can be parallel to the m/z and intensity values.
#         relevant_fragments = np.array([''] * len(mz_values), dtype=object)  # Initialize an array for relevant fragments
#         num_found_fragments = 0
#         for mz_frag_pair in fragment_data:
#             frag_mz = float(mz_frag_pair[0])
#             mz_idx = np.where(mz_values == frag_mz)[0]
#             if len(mz_idx):
#                 relevant_fragments[mz_idx[0]] = mz_frag_pair[1]
#                 num_found_fragments += 1
#         if num_found_fragments != len(mz_values):
#             warnings.warn(f"Not all fragments were found in the m/z values for compound {cmpd_i}! Skipping compound")
#             continue  # Skip this compound if not all fragments were found
#         if np.any(relevant_fragments == ''):
#             warnings.warn(f"Some fragments were not found in the m/z values for compound {cmpd_i}! Skipping compound")
#             continue

#         # IBM fragments are given by SMILES -- convert to molecular formulae
#         fragment_formulas = list()  # List to store the formulas of the fragments
#         for frag in relevant_fragments:
#             mol = Chem.MolFromSmiles(frag, sanitize=False) # Ignore sanitization because fragments are ions with incorrect valence at times
#             mol.UpdatePropertyCache(strict=False) # Update the property cache to ensure the formula is correct
#             frag_formula = CalcMolFormula(mol)
#             fragment_formulas.append(frag_formula)
#         # Get rid of any ion tokens
#         fragment_formulas = [frag.replace('+', '').replace('-', '') for frag in fragment_formulas]
#         # Create the required data dictionary for the current compound
#         req_data = getRequiredFields()
#         # DiffMS sorts by intensity, so we will do the same
#         sort_idxs = np.argsort(intensity_values)
#         req_data['mz'] = mz_values[sort_idxs]
#         req_data['intensity'] = intensity_values[sort_idxs]
#         req_data['fragments'] = np.array(fragment_formulas)[sort_idxs]
#         req_data['formula'] = formula
#         req_data['smiles'] = smiles
#         req_data['ion'] = ion
#         req_data = extrapolateCurrentData(req_data)  # Extrapolate the data for the current compound
#         gathered_data.append(req_data)

#     yield from gathered_data  # Yield the extracted data for the each compound


def extractIbmChunkData(chunk: pd.DataFrame, ion_modes: str, energies: int):
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
    num_chunk_cmpds = len(chunk)
    for cmpd_i in tqdm(range(num_chunk_cmpds), desc='Extracting data from chunk'):
        for ion_mode in ion_modes:
            for energy in energies:
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
                if np.any(relevant_fragments == ''):
                    warnings.warn(f"Some fragments were not found in the m/z values for compound {cmpd_i}! Skipping compound")
                    continue

                # IBM fragments are given by SMILES -- convert to molecular formulae
                fragment_formulas = list()  # List to store the formulas of the fragments
                for frag in relevant_fragments:
                    mol = Chem.MolFromSmiles(frag, sanitize=False) # Ignore sanitization because fragments are ions with incorrect valence at times
                    mol.UpdatePropertyCache(strict=False) # Update the property cache to ensure the formula is correct
                    frag_formula = CalcMolFormula(mol)
                    fragment_formulas.append(frag_formula)
                # Get rid of any ion tokens
                fragment_formulas = [frag.replace('+', '').replace('-', '') for frag in fragment_formulas]
                # Create the required data dictionary for the current compound
                req_data = getRequiredFields()
                # DiffMS sorts by intensity, so we will do the same
                sort_idxs = np.argsort(intensity_values)
                req_data['mz'] = mz_values[sort_idxs]
                req_data['intensity'] = intensity_values[sort_idxs]
                req_data['fragments'] = np.array(fragment_formulas)[sort_idxs]
                req_data['formula'] = formula
                req_data['smiles'] = smiles
                req_data['ion'] = ion
                req_data = extrapolateCurrentData(req_data)  # Extrapolate the data for the current compound
                yield req_data


def extractIbmData(cfg, dataset_cfg):
    """
    Extracts the IBM dataset from the specified directory and returns the relevant data.
    Assumes the only files within the directory are the IBM dataset files.
    Args:
        cfg (dict): Configuration dictionary containing parameters like ion_mode and energy.
    Returns:
        stored_data: A dictionary containing the extracted data for each compound.
    """
    import pyarrow.parquet as pq

    data_dir = cfg.read_dir

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

    file_writer = DiffMSFileWriter(dataset_name='ibm',
                                   total_cmpds=num_total_cmpds,
                                   save_dir=dataset_cfg['save_dir'],
                                   remove_hydrogen=dataset_cfg['remove_hydrogen'])

    # Iterate through each file and extract the data using extractIbmChunkData generator
    for file in tqdm(all_files, desc='Extracting data from files'):
        data = pd.read_parquet(file, engine='pyarrow')
        # data_generator = extractIbmChunkData(data, ion_mode=cfg.ion_mode, energy=cfg.energy)
        data_generator = extractIbmChunkData(data, ion_modes=cfg.ion_mode, energies=cfg.energy)
        for cmpd_data in data_generator:
            file_writer.createCompundDiffMsFiles(cmpd_data)

    file_writer.createDatasetDiffMsFiles(cfg.split_file, cfg.split_name, cfg.labels_name)
    # After all files are processed, write the files that require the entire dataset to be processed
    if dataset_cfg['cleanup']:
        file_writer.cleanupHelperFiles()
#####################################################################################################################
# END IBM data extraction specific functions


@hydra.main(version_base='1.1', config_path='../configs/extract_dataset', config_name='main')
def main(cfg):
    # Make sure read and save directories are provided
    assert cfg.dataset.read_dir is not None, "Please provide a read directory for the dataset!"
    assert cfg.dataset.save_dir is not None, "Please provide a save directory for the dataset!"

    # Ensure the save directory exists
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)

    spectrum_dir = subformula_dir = helper_dir = None

    dataset_cfg = {'spec_dir': spectrum_dir, 
                   'subformula_dir': subformula_dir,
                   'helper_dir': helper_dir,
                   'save_dir': cfg.dataset.save_dir,
                   'remove_hydrogen': cfg.dataset.remove_hydrogen,
                   'cleanup': cfg.cleanup}
    
    if cfg.dataset.dataset_name.lower() == 'ibm':
        extractIbmData(cfg.dataset, dataset_cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.dataset_name} is not supported yet!")
    

if __name__ == '__main__':
    main()