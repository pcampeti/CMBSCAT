# Map-based emulator for CMB systematics with scattering covariances


`cmbscat` is a pip-installable python package that can synthesize new full-sky map samples (*emulations*) on the HEALPix sphere which are both visually and statistically similar to the ones found in an (eventually small) dataset of simulations. This *emulator* is described in detail in the accompanying paper [arxiv:2503.11643](https://arxiv.org/abs/2503.11643).  

`cmbscat` relies heavily on the [`HealpixML`](https://github.com/jmdelouis/HealpixML) library for efficient scattering covariance computation on the HEALPix sphere. 

## Install with pip
You can install it simply with:
```
pip install cmbscat
```
We recommend using `python 3.11`.

Alternatively, you can install from source with:
```
git clone https://github.com/pcampeti/CMBSCAT.git
cd CMBSCAT/
pip install -r requirements.txt
pip install -e .
```

## Usage

You can generate a new dataset of CMB systematics maps by doing:

```python
from cmbscat import cmbscat_pipe

# Set emulator parameters
params = {
    'NNN'          : 10,               # Number of input reference maps
    'gauss_real'   : True,             # Generate new input data as Gaussian realizations from pixel covariance of original data
    'NGEN'         : 10,               # Batch size for gradient descent
    'n_samples'    : 10,               # Samples in the input dataset
    'nside'        : 16,               # N_side of input maps
    'NORIENT'      : 4,                # Orientations in the SC
    'nstep'        : 50,               # Steps in gradient descent
    'KERNELSZ'     : 3,                # Wavelet kernel size
    'outname'      : 'example',        # Output name
    'outpath'      : './data/',        # Output path
    'data'         : 'variable_gain_sims.npy'  # Input data path (assuming you run from inside the /CMBSCAT/notebook/ folder)
}

# Initialize pipeline...
pipeline = cmbscat_pipe(params)

#...and run! This generates NGEN new maps for each of the n_samples input maps
pipeline.run()
```

## Tutorial Notebook
You can find an introductory notebook explaining all features of the `cmbscat` package [here](https://github.com/pcampeti/CMBSCAT/blob/main/notebook/CMBSCAT_demo.ipynb). 

Specifically, we apply it to simulated maps of an example of CMB satellites instrumental systematics, as described in the accompanying paper [arxiv:2503.11643](https://arxiv.org/abs/2503.11643).

<img width="1066" alt="image" src="https://github.com/user-attachments/assets/aa976368-5d3c-4879-a461-3d0cd120ba4a" />

## Citations
Should this code be used in any way, we kindly ask that the following article is cited:

```
@article{campeti:systematics_emulator, 
   author      = "Paolo Campeti, Jean-Marc Delouis, Luca Pagano, Erwan Allys, Massimiliano Lattanzi, Martina Gerbino",
   title       = "From few to many maps: A fast map-level emulator for extreme augmentation of CMB systematics datasets",
   eprint = "2503.11643",
   archivePrefix = "arXiv",
   primaryClass = "astro-ph.CO",
   month = "Mar",
   year = "2025"
}
```
