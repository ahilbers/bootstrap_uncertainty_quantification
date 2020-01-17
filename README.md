# 2020_bootstrap_uncertainty_quantification
Data, model files and example code related to the paper "Quantifying demand and weather uncertainty in power system models using the *m* out of *n* bootstrap".

The file `main.py` can be run directly from a command line and provides an example application of the *bootstrap uncertainty quantification* algorithm. It creates a new directory called `outputs` with the point estimates and standard deviation estimates for different model outputs. Arguments can be specified directly in the script.

This repository contains each of the three models discussed in the paper. The `1region_cont`, `6regions_cont` and `6regions_disc` models correspond to the *1-region LP*, *6-region LP* and *6-region MILP* models respectively in the paper.

The models used in this paper come from a larger class of test power system models, which are available as open-source software in [this repository](https://ahilbers.github.io), where they are documented and available in a more general form. If you want to use these models for your own research, its easier to use that respository instead of this one.




## Contains

### Modelling & data files

- `models/`: power system model generating files, for `Calliope` (see acknowledgements). The `demand_wind.csv` files present under `timeseries_data` in each model are just placeholders used to initialise the model, and the correct data is loaded in later.
- `data/`: demand and weather time series data
  - `demand_wind_1region.csv`: demand and wind time series used in *1-region LP* model in paper
  - `demand_wind_6regions.csv`: demand and wind time series used in *6-region LP* and *6-region MILP* models in paper


### Code

- `samplers.py`: functions for bootstrap sampling of the demand & wind time series: both the *months* and *weeks* scheme from the paper
- `model_runs.py`: functions that perform power system model runs
- `main.py`: a script that performs one full run through the methodology, using a single long simulation for a point estimate and multiple short simulations across bootstrap samples to estimate the standard deviation. It can be run directly from a command line.




## Requirements & Installation

Since `main.py`, containing all code, is a short file with only a few functions, it's probably easier to directly copy-paste any relevant code into a personal project as opposed to installing a new module. For this reason, this repository does not contain a `setup.py` file.

Running `main.py` requires:
- Python modules:
  - `Calliope 0.6.4`:  see [this link](https://calliope.readthedocs.io/en/stable/user/installation.html) for installation.
  - `numpy 1.62.2`
  - `pandas 0.24.2`
- Other:
  - `cbc`: open-source optimiser: see [this link](https://projects.coin-or.org/Cbc) for installation. Other solvers (e.g. `gurobi`) are also possible -- the solver can be specified in `models/{MODEL_NAME}/model.yaml`.





## Contact

Adriaan Hilbers. Department of Mathematics, Imperial College London. [a.hilbers17@imperial.ac.uk](mailto:a.hilbers17@imperial.ac.uk).




## Acknowledgements

Models are constructed in the modelling framework `Calliope`, created by Stefan Pfenninger and Bryn Pickering. See [callio.pe](https://callio.pe) or the following paper for details:

- Pfenninger, S. and Pickering, B. (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:[10.21105/joss.00825](https://doi.org/10.21105/joss.00825).

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following paper:

- Bloomfield, H. C., Brayshaw, D. J. and Charlton-Perez, A. (2019) Characterising the winter meteorological drivers of the European electricity system using Targeted Circulation Types. Meteorological Applications. ISSN 1469-8080 (In Press). doi:[10.1002/met.1858](https://doi.org/10.1002/met.1858)
