[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# bootstrap_uncertainty_quantification

## Summary

This repository contains example code related to the paper "Efficient quantification of the impact of demand and weather uncertainty in power system models".

It also contains all source code and data associated with the three test-case models used in the paper (the *LP planning*, *MILP planning* and *operation* models). These models are modified versions from a more general class of test power system models, available as open-source software in [this repository](https://github.com/ahilbers/renewable_test_PSMs), where they are documented and available in a more general form. If you want to use these models for your own research, its easier to use that respository instead of this one.

**Note**: In a previous iteration, this paper was called "Quantifying demand and weather uncertainty in power system models using the *m* out of *n* bootstrap, of which a preprint is available on arXiv [here](https://arxiv.org/abs/1912.10326). The models have changed slightly since that version. If you've come from that preprint, check out release v1.0.0 of this repository.




## How to cite

If you use this code in your own research, please cite the following paper:

- Hilbers, A.P., Brayshaw, D.J., Gandy, A. (2019). Quantifying demand and weather uncertainty in power system models using the m out of n bootstrap. [arXiv:1912.10326](https://arxiv.org/abs/1912.10326).


Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:[10.21105/joss.00825](https://doi.org/10.21105/joss.00825).





## Usage

To run an example of the methodology, call

```
python3 main.py
```

from a command line. This runs a very simple example of the methodology. To customise it, it's easiest to change arguments directly in `main.py` -- the settings can be specified in the function `run_example`. It creates a new directory called `outputs` with the point estimates and standard deviation estimates for the outputs of the `LP_planning` model, run across 2017 data.

The default settings use short samples, just to see if the methodology is working. If you want to actually use the method, it's recommended to increase the subsample length and number of subsamples. This can be done by changing the arguments in the `run_example` function in `main.py`.

To keep things simple and clear, everything runs in series. If you'd like to see the full scale code, where the bootstrap runs can be performed in parallel on a computing cluster, email [Adriaan Hilbers](mailto:a.hilbers17@imperial.ac.uk).






## Contains

### Model & data files

- `models/`: power system model generating files, for `Calliope` (see acknowledgements).
- `data/`: demand and weather time series data


### Code

- `main.py`: a script that performs one full run through the methodology, using a single long simulation for a point estimate and multiple short simulations across bootstrap samples to estimate the standard deviation. It can be called from a command line.
- `samplers.py`: functions for bootstrap sampling of the demand & wind time series: both the *months* and *weeks* scheme from the paper
- `models.py`: some utility code for the models
- `tests.py`: some tests to check if the models are behaving as expected.




## Requirements & Installation

Since `main.py`, containing all code, is a short file with only a few functions, it's probably easier to directly copy-paste any relevant code into a personal project as opposed to installing a new module. For this reason, this repository does not contain a `setup.py` file.

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

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following paper:

- Bloomfield, H. C., Brayshaw, D. J. and Charlton-Perez, A. (2019) Characterising the winter meteorological drivers of the European electricity system using Targeted Circulation Types. Meteorological Applications. ISSN 1469-8080 (In Press). doi:[10.1002/met.1858](https://doi.org/10.1002/met.1858)
