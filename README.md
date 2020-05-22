[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# bootstrap_uncertainty_quantification

## Summary

This repository contains example code related to the paper "Efficient quantification of the impact of demand and weather uncertainty in power system models".

It also contains all source code and data associated with the three test-case models used in the paper (the *LP planning*, *MILP planning* and *operation* models). These models are modified versions of a more general class of test power system models, available open-source in [this repository](https://github.com/ahilbers/renewable_test_PSMs), where they are documented and available in a more general form. If you want to use these models for your own research, its easier to use that respository instead of this one.





## Usage

To run an example of the methodology, call

```
python3 main.py
```

from a command line. This runs a simple example of the BUQ algorithm on the *LP_planning* model. The default settings take 10-15 minutes to run. To customise it, it's easiest to change arguments directly in `main.py` -- the settings can be specified in the function `run_example`. It creates a new directory called `outputs` with the point estimates and standard deviation estimates for the outputs of the `LP_planning` model, run across 2017 data.

The default settings use short samples to run quickly. If you want to actually use the method, it's recommended to increase the subsample length and number of subsamples. This can be done by changing the arguments in the `run_example` function in `main.py`.

This repository contains a few tests and benchmarks which can be used to check if the code is running as expected. Running `tests.py` from a command line starts a number of consistency tests and checks the outputs from a very simple application fo the BUQ algorithm against a set of benchmarks. It should take around 10-15 minutes to run, and will log whether all tests pass.





## Contains

### Model & data files

- `models/`: power system model generating files, for `Calliope` (see acknowledgements).
- `data/`: demand and weather time series data
- `test_benchmarks`: some benchmarks -- used by `tests.py` to see if things are working correctly.


### Code

- `main.py`: a script that performs one full run through the methodology, using a single long simulation for a point estimate and multiple short simulations across bootstrap samples to estimate the standard deviation. It can be called from a command line.
- `buq.py`: functions for the bootstrap uncertainty quantification (BUQ) algorithm, both the *months* and *weeks* scheme from the paper
- `models.py`: some utility code for the models
- `tests.py`: some tests to check if the models are behaving as expected.




## Requirements & Installation

Since `main.py`, containing all code, is a short file with only a few functions, and is dependent on a particularl model framework, it's probably easier to fork and edit any relevant code into a personal project as opposed to installing a new module. For this reason, this repository does not contain a `setup.py` file.

Running `main.py` works with:
- Python modules:
  - `Calliope 0.6.5`:  see [this link](https://calliope.readthedocs.io/en/stable/user/installation.html) for installation.
  - Basic modules: `numpy`, `pandas`.
- Other:
  - `cbc`: open-source optimiser: see [this link](https://projects.coin-or.org/Cbc) for installation. Other solvers (e.g. `gurobi`) are also possible -- the solver can be specified in `models/6_region/model.yaml`.
All code is known to run with the above setup, but may also run with different verions than those specified above.





## Contact

[Adriaan Hilbers](https://ahilbers.github.io). Department of Mathematics, Imperial College London. [a.hilbers17@imperial.ac.uk](mailto:a.hilbers17@imperial.ac.uk).




## Acknowledgements

Models are constructed in the modelling framework `Calliope`, created by Stefan Pfenninger and Bryn Pickering. See [callio.pe](https://callio.pe) or the following paper for details:

- Pfenninger, S. and Pickering, B. (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:[10.21105/joss.00825](https://doi.org/10.21105/joss.00825).

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following paper and dataset:

- Bloomfield, H. C., Brayshaw, D. J. and Charlton-Perez, A. (2019) Characterising the winter meteorological drivers of the European electricity system using Targeted Circulation Types. Meteorological Applications. ISSN 1469-8080 (In Press). doi:[10.1002/met.1858](https://doi.org/10.1002/met.1858)

- HC Bloomfield, DJ Brayshaw, A Charlton-Perez (2020). MERRA2 derived time series of European country-aggregate electricity demand, wind power generation and solar power generation. University of Reading. Dataset. doi:[10.17864/1947.239](https://doi.org/10.17864/1947.239)
