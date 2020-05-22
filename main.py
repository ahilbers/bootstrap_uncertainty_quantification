"""
Run example application of bootstrap uncertainty quantification
algorithm. Arguments can be specified in the function 'run_example'
"""


import os
import logging
import buq


def run_example():
    """Run an example of the methodology.

    Arguments can be specified below. Notes:
    - model_name_in_paper: 'LP_planning', 'MILP_planning' or 'operation'
    - num_blocks_per_bin: depends on the bootstrap scheme:
      - 'months': number of times each week is sampled, so that the total
        subsample size is (365*num_blocks_per_bin) days.
      - 'weeks': number of weeks from each season sampled, so that the
        total subsample size is (28*num_blocks_per_bin) days.
    """

    # Arguments -- change as desired, see notes above
    model_name_in_paper = 'LP_planning'
    point_estimate_range = [2017, 2017]   # Includes endpoints
    bootstrap_scheme = 'weeks'
    num_blocks_per_bin = 3
    num_bootstrap_samples = 30    # K in paper
    logging_level = 'INFO'   # use 'ERROR' for fewer logging statements

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=getattr(logging, logging_level),
        datefmt='%Y-%m-%d,%H:%M:%S'
    )

    if os.path.exists('outputs'):
        raise AssertionError(
            'Simulations will save outputs to a directory called '
            '`outputs`, but this already exists. Delete or rename '
            'that directory.'
        )

    # Run the methodology, return point estimates and stdev estimates
    results = buq.calculate_point_estimate_and_stdev(
        model_name_in_paper=model_name_in_paper,
        point_estimate_range=point_estimate_range,
        bootstrap_scheme=bootstrap_scheme,
        num_blocks_per_bin=num_blocks_per_bin,
        num_bootstrap_samples=num_bootstrap_samples
    )

    # Save outputs to CSV
    logging.info('Done with all model runs. '
                 'Saving outputs to new directory `outputs`')
    os.mkdir('outputs')
    results.to_csv('outputs/model_outputs.csv', float_format='%.5f')


if __name__ == '__main__':
    run_example()
