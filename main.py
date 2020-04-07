"""
Run example application of bootstrap uncertainty quantification
algorithm. Arguments can be specified in this script.

Warnings saying
    `monetary` interest rate of zero for technology XXXX, setting
    depreciation rate as 1/lifetime.
may appear. This is a result of the choice of model setup and can be ignored.
"""


import os
import calliope
import model_runs


def run_example():
    """Run an example of the methodology.

    Arguments can be specified below. Notes:
    - For model_name_in_paper, the possibilities are 1_region_LP,
      6_region_LP and 6_region_MILP
    - num_blocks_per_bin: for the 'months' scheme, this is the number of
      each calendar month sampled. For the 'weeks' scheme, this
      is the number of weeks sampled from each meteorological season
    """

    # Arguments
    model_name_in_paper = '1_region_LP'
    point_estimate_range = [2017, 2017]   # Includes endpoints
    bootstrap_scheme = 'weeks'
    num_blocks_per_bin = 2
    num_bootstrap_samples = 30

    # Run the methodology, return point estimates and stdev estimates
    results = model_runs.calculate_point_estimate_and_stdev(
        model_name_in_paper=model_name_in_paper,
        point_estimate_range=point_estimate_range,
        bootstrap_scheme=bootstrap_scheme,
        num_blocks_per_bin=num_blocks_per_bin,
        num_bootstrap_samples=num_bootstrap_samples
        )

    # Save outputs to CSV
    os.mkdir('outputs')
    results.to_csv('outputs/model_outputs.csv', float_format='%.5f')


if __name__ == '__main__':
    run_example()
