# 2020_bootstrap_uncertainty_quantification
Data, model files and example code related to the paper "Quantifying demand and weather uncertainty in power system models using the *m* out of *n* bootstrap".




## Contains

### Modelling & data files

- `model_files/`: power system model generating files, for `Calliope` (see acknowledgements)
- `data/`: demand and weather time series data
  - `demand_wind_1region.csv`: demand and wind time series used in *1-region LP* model in paper
  - `demand_wind_6regions.csv`: demand and wind time series used in *6-region LP* and *6-region MILP* models in paper


### Code

- `main.py`: XXXX XXXX XXXX XXXX




## Requirements & Installation

Since `main.py`, containing all code, is a short file with only a few functions, it's probably easier to directly copy-paste any relevant code into a personal project as opposed to installing a new module. For this reason, this repository does not contain a `setup.py` file.

Running `main.py` requires:
- Python modules:
  - `Calliope 0.64`:  see [this link](https://calliope.readthedocs.io/en/stable/user/installation.html) for installation. By default, `Calliope` is installed in a virtual environment, which we assume is called `calliope`.
  - `numpy 1.62.2`
  - `pandas 0.24.2`
- Other:
  - `cbc`: open-source optimiser: see [this link](https://projects.coin-or.org/Cbc) for installation. Other solvers (e.g. `gurobi`) are also possible -- the solver can be specified in `model_files/model.yaml`.





## Contact

Adriaan Hilbers. Department of Mathematics, Imperial College London. [a.hilbers17@imperial.ac.uk](mailto:a.hilbers17@imperial.ac.uk).




## Acknowledgements

Models are constructed in the modelling framework `Calliope`, created by Stefan Pfenninger and Bryn Pickering. See [callio.pe](callio.pe) or the following paper for details:

- Pfenninger, S. and Pickering, B. (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:[10.21105/joss.00825](doi.org/10.21105/joss.00825).

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following paper:

Bloomfield, H. C., Brayshaw, D. J. and Charlton-Perez, A. (2019) Characterising the winter meteorological drivers of the European electricity system using Targeted Circulation Types. Meteorological Applications. ISSN 1469-8080 (In Press). doi:[10.1002/met.1858](doi.org/10.1002/met.1858)
