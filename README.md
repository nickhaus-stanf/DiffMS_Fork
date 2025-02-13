# DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra

This is the codebase for our preprint DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra.

The DiffMS codebase is adapted from [DiGress](https://github.com/cvignac/DiGress). 

## Environment installation
This code was tested with PyTorch 2.3.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a conda environment with rdkit:
    
    ```conda create -y -c conda-forge -n diffms rdkit=2024.09.4 python=3.9```
  - `conda activate diffms`
    
  - Install a corresponding version of pytorch, for example: 
    
    ```pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118```

  - Run:
    
    ```pip install -e .```


## Run the code
  
For fingerprint-molecule pretraining run [fp2mol_main.py](src/fp2mol_main.py). The primary pretraining dataset in our paper is referred to as 'combined' in the fp2mol.yaml config. 

To finetune the end-to-end model on spectra-molecule generation, run [fp2mol_main.py](src/spec2mol_main.py) and select the spectra dataset by modifying the dataset property in configs/config.yaml to canopus (NPLIB1) or msg (MassSpecGym)

We rely on the [MIST Codebase](https://github.com/samgoldman97/mist) for pretraining the spectra encoder.

## License

DiffMS is released under the [MIT](LICENSE.txt) license.

## Contact

If you have any questions, please reach out to mbohde@tamu.edu