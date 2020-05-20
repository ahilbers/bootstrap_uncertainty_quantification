"""
Run example application of bootstrap uncertainty quantification
algorithm. Arguments can be specified in the function 'run_example'
"""


import os
import time
import logging
import calliope
import numpy as np
import pandas as pd
import samplers
import models
import tests


def import_time_series_data():
    """Import time series data for model, without any time slicing."""
    ts_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    return ts_data


def run_simulation(model_name_in_paper, ts_data, run_id=0):
    """Run Calliope model with demand & wind data.

    Parameters:
    -----------
    model_name_in_paper (str) : 'LP_planning', 'MILP_planning' or
        'operate'
    ts_data (pandas DataFrame) : demand & wind time series data
    run_id (int or str) : unique id, useful if running in parallel

    Returns:
    --------
    results: pandas DataFrame with model outputs
    """

    start = time.time()
    if model_name_in_paper == 'LP_planning':
        model = models.SixRegionModel(ts_data=ts_data,
                                      run_mode='plan',
                                      baseload_integer=False,
                                      baseload_ramping=False,
                                      allow_unmet=True,
                                      run_id=run_id)
    elif model_name_in_paper == 'MILP_planning':
        model = models.SixRegionModel(ts_data=ts_data,
                                      run_mode='plan',
                                      baseload_integer=True,
                                      baseload_ramping=False,
                                      allow_unmet=True,
                                      run_id=run_id)
    elif model_name_in_paper == 'operation':
        model = models.SixRegionModel(ts_data=ts_data,
                                      run_mode='operate',
                                      baseload_integer=False,
                                      baseload_ramping=True,
                                      allow_unmet=True,
                                      run_id=run_id)
    else:
        raise ValueError('Invalid model name.')

    # Run model and save results
    model.run()
    finish = time.time()
    tests.test_output_consistency_6_region(model, run_mode=(
        'operate' if model_name_in_paper == 'operation' else 'plan'
    ))
    results = model.get_summary_outputs()
    results.loc['time'] = finish - start

    return results


def run_years_simulation(model_name_in_paper, startyear, endyear, run_id=0):
    """Run Calliope model with certain years of data."""

    ts_data = import_time_series_data()
    ts_data = ts_data.loc[str(startyear):str(endyear)]
    results = run_simulation(model_name_in_paper, ts_data=ts_data,
                             run_id=run_id)

    return results


def run_bootstrap_simulation(model_name_in_paper, scheme,
                             num_blocks_per_bin, run_id=0):
    """Run model with bootstrap sampled data

    Parameters:
    -----------
    model_name_in_paper (str) : 'LP_planning', 'MILP_planning' or
        'operate'
    scheme: either 'months' or 'weeks' -- scheme used to create bootstrap
        samples
    num_blocks_per_bin: either the number of months sampled from each
        calendar month, or the number of weeks sampled from each season

    Returns:
    --------
    results: pandas DataFrame with model outputs
    """

    ts_data = import_time_series_data()

    # Create bootstrap sample and run model
    if scheme == 'months':
        sample = samplers.bootstrap_sample_months(ts_data,
                                                  num_blocks_per_bin)
    elif scheme == 'weeks':
        sample = samplers.bootstrap_sample_weeks(ts_data,
                                                 num_blocks_per_bin)
    else:
        raise ValueError('Must be either months or weeks scheme')
    results = run_simulation(model_name_in_paper, ts_data=sample,
                             run_id=run_id)

    return results


def run_buq_algorithm(model_name_in_paper,
                      point_sample_length,
                      bootstrap_scheme,
                      num_blocks_per_bin,
                      num_bootstrap_samples):
    """Run through BUQ algorithm once to estimate standard deviation.

    Parameters:
    -----------
    model_name_in_paper (str) : 'LP_planning', 'MILP_planning' or
        'operate'
    point_sample_length (int) : length of sample used to determine point
        estimate (in hours), used only for rescaling
    boostrap scheme (str) : bootstrap scheme for calculating standard
        deviation: 'months' or 'weeks'
    num_blocks_per_bin (int) : number of months from each calendar month
        or number of weeks from each season
    num_bootstrap_samples (int) : number of bootstrap samples over which to
        calculate the standard deviation

    Returns:
    --------
    point_estimate_stdev (pandas DataFrame) : estimates for the standard
        deviation of each model output
    """

    if bootstrap_scheme == 'weeks':
        bootstrap_sample_length = num_blocks_per_bin * 4 * 7 * 24
    elif bootstrap_scheme == 'months':
        bootstrap_sample_length = num_blocks_per_bin * 8760

    # Calculate variance across bootstrap samples
    logging.info('Calculating stdev estimate...')
    logging.info('Starting bootstrap samples')

    # Run model for each bootstrap sample
    run_index = np.arange(num_bootstrap_samples)
    for sample_num in run_index:
        logging.info('Calculating bootstrap sample %s', sample_num+1)
        results = run_bootstrap_simulation(model_name_in_paper,
                                           bootstrap_scheme,
                                           num_blocks_per_bin)
        if sample_num == 0:
            outputs = pd.DataFrame(columns=np.arange(num_bootstrap_samples),
                                   index=results.index)
        outputs[sample_num] = results.loc[:, 'output']
        logging.info('Done.')

    # Calculate variance across model outputs
    bootstrap_variance = outputs.var(axis=1)

    # Rescale variance to determine stdev of point estimate
    point_estimate_variance = (
        (bootstrap_sample_length/point_sample_length) * bootstrap_variance
    )
    point_estimate_stdev = pd.DataFrame(np.sqrt(point_estimate_variance),
                                        columns=['stdev'])
    logging.info('Done calculating stdev estimate.')

    return point_estimate_stdev


def calculate_point_estimate_and_stdev(model_name_in_paper,
                                       point_estimate_range,
                                       bootstrap_scheme,
                                       num_blocks_per_bin,
                                       num_bootstrap_samples):
    """Calculate point estimate using a single long simulation and estimate
    standard deviation using multiple short simulations and BUQ algorithm.

    Parameters:
    -----------
    model_name_in_paper (str) : 'LP_planning', 'MILP_planning' or
        'operate'
    point_estimate_range (list) : range of years over which to calculate
        point estimate, e.g. [2017, 2017] for just the year 2017 (includes
        endpoints).
    boostrap scheme (str) : bootstrap scheme for calculating standard
        deviation: 'months' or 'weeks'
    num_blocks_per_bin (int) : number of months from each calendar month
        or number of weeks from each season
    num_bootstrap_samples (int) : number of bootstrap samples over which to
        calculate the standard deviation

    Returns:
    --------
    estimate_with_stdev (pandas DataFrame) : has 2 columns: the point
        estimates and the stdev of the relevant model outputs
    """

    point_sample_length = 8760 * (point_estimate_range[1]
                                  - point_estimate_range[0] + 1)

    # Calculate point estimate via single long simulation
    logging.info('Calculating point estimate...')
    point_estimate = run_years_simulation(
        model_name_in_paper=model_name_in_paper,
        startyear=point_estimate_range[0],
        endyear=point_estimate_range[1]
    )
    point_estimate = pd.DataFrame(point_estimate.values,
                                  columns=['point_estimate'],
                                  index=point_estimate.index)
    logging.info('Done calculating point_estimate.')


    # Estimate standard deviation with BUQ algorithm
    point_estimate_stdev = run_buq_algorithm(
        model_name_in_paper=model_name_in_paper,
        point_sample_length=point_sample_length,
        bootstrap_scheme=bootstrap_scheme,
        num_blocks_per_bin=num_blocks_per_bin,
        num_bootstrap_samples=num_bootstrap_samples
    )
    point_estimate_stdev = pd.DataFrame(point_estimate_stdev.values,
                                        columns=['stdev'],
                                        index=point_estimate_stdev.index)

    # Create single dataframe with point and standard deviation estimate
    estimate_with_stdev = point_estimate.join(point_estimate_stdev)

    return estimate_with_stdev


def run_example():
    """Run an example of the methodology.

    Arguments can be specified below. Notes:
    - For model_name_in_paper, the possibilities are 1_region_LP,
      6_region_LP and 6_region_MILP
    - num_blocks_per_bin: depends on the bootstrap scheme:
      - 'months': number of times each week is sampled, so that the total
        subsample size is (365*num_blocks_per_bin) days.
      - 'weeks': number of weeks from each season sampled, so that the
        total subsample size is (28*num_blocks_per_bin) days.
    """

    # Arguments -- change as desired, see notes above
    model_name_in_paper = 'operation'
    point_estimate_range = [2017, 2017]   # Includes endpoints
    bootstrap_scheme = 'weeks'
    num_blocks_per_bin = 1
    num_bootstrap_samples = 2    # K in paper
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

    np.random.seed(42)    # Useful for checking against benchmark

    # Run the methodology, return point estimates and stdev estimates
    results = calculate_point_estimate_and_stdev(
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
