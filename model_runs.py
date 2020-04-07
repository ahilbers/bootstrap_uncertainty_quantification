"""Power system model runs, using Calliope framework."""


import time
import numpy as np
import pandas as pd
import samplers
import models
import calliope


def import_time_series_data(model_name_in_paper):
    """Import time series data for model, without any time slicing."""
    if '1_region' in model_name_in_paper:
        ts_data_path = 'data/demand_wind_1_region.csv'
    elif '6_region' in model_name_in_paper:
        ts_data_path = 'data/demand_wind_6_region.csv'
    else:
        raise NotImplementedError()
    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    return ts_data


def run_simulation(model_name_in_paper, ts_data, save_csv=False):
    """Run Calliope model with demand & wind data.

    Parameters:
    -----------
    model_name_in_paper (str) : '1_region_LP', '6_region_LP'
        or '6_region_MILP'
    ts_data (pandas DataFrame) : demand & wind time series data
    save_csv (bool) : save larger set of outputs (the full Callope
        set of outputs) as separate CSV files (insert directory name)

    Returns:
    --------
    results: pandas DataFrame with model outputs
    """

    start = time.time()

    if model_name_in_paper == '1_region_LP':
        model = models.OneRegionModel(ts_data=ts_data,
                                      run_mode='plan',
                                      baseload_integer=False,
                                      baseload_ramping=False,
                                      allow_unmet=True)
    elif model_name_in_paper == '6_region_LP':
        model = models.SixRegionModel(ts_data=ts_data,
                                      run_mode='plan',
                                      baseload_integer=False,
                                      baseload_ramping=False,
                                      allow_unmet=True)
    elif model_name_in_paper == '6_region_MILP':
        model = models.SixRegionModel(ts_data=ts_data,
                                      run_mode='plan',
                                      baseload_integer=True,
                                      baseload_ramping=True,
                                      allow_unmet=True)
    else:
        raise NotImplementedError()

    # Run model and save results
    model.run()
    if save_csv is not False:
        model.to_csv(save_csv)
    finish = time.time()
    results = model.get_summary_outputs()
    results.loc['time'] = finish - start

    return results


def run_years_simulation(model_name_in_paper, startyear, endyear):
    """Run Calliope model with certain years of data.

    Parameters:
    -----------
    model_name_in_paper (str) : '1_region_LP', '6_region_LP'
        or '6_region_MILP'
    startyear: first year of data to run model on
    lastyear: last year of data to run model on

    Returns:
    --------
    results: pandas DataFrame with model outputs
    """

    ts_data = import_time_series_data(model_name_in_paper)
    ts_data = ts_data.loc[str(startyear):str(endyear)]
    results = run_simulation(model_name_in_paper, ts_data=ts_data)
    return results


def run_bootstrap_simulation(model_name_in_paper, scheme,
                             num_blocks_per_bin):
    """Run model with bootstrap sampled data

    Parameters:
    -----------
    model_name_in_paper (str) : '1_region_LP', '6_region_LP'
        or '6_region_MILP'
    scheme: either 'months' or 'weeks' -- scheme used to create bootstrap
        samples
    num_blocks_per_bin: either the number of months sampled from each
        calendar month, or the number of weeks sampled from each season

    Returns:
    --------
    results: pandas DataFrame with model outputs
    """

    ts_data = import_time_series_data(model_name_in_paper)

    # Create bootstrap sample and run model
    if scheme == 'months':
        sample = samplers.bootstrap_sample_months(ts_data, num_blocks_per_bin)
    elif scheme == 'weeks':
        sample = samplers.bootstrap_sample_weeks(ts_data, num_blocks_per_bin)
    else:
        raise ValueError('Must be either months or weeks scheme')
    results = run_simulation(model_name_in_paper, ts_data=sample)
    return results


def run_buq_algorithm(model_name_in_paper,
                      point_sample_length,
                      bootstrap_scheme,
                      num_blocks_per_bin,
                      num_bootstrap_samples):
    """Run through BUQ algorithm once to estimate standard deviation.

    Parameters:
    -----------
    model_name_in_paper (str) : '1_region_LP', '6_region_LP'
        or '6_region_MILP'
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
    print('Calculating stdev estimate...')
    print('Starting bootstrap samples')

    # Run model for each bootstrap sample
    run_index = np.arange(num_bootstrap_samples)
    for sample_num in run_index:
        print('Calculating bootstrap sample {}...'.format(sample_num+1))
        results = run_bootstrap_simulation(model_name_in_paper,
                                           bootstrap_scheme,
                                           num_blocks_per_bin)
        if sample_num == 0:
            outputs = pd.DataFrame(columns=np.arange(num_bootstrap_samples),
                                   index=results.index)
        outputs[sample_num] = results.loc[:, 'output']
        print('Done.')

    # Calculate variance across model outputs
    bootstrap_variance = outputs.var(axis=1)

    # Rescale variance to determine stdev of point estimate
    point_estimate_variance = (
        (bootstrap_sample_length/point_sample_length) * bootstrap_variance
    )
    point_estimate_stdev = pd.DataFrame(np.sqrt(point_estimate_variance),
                                        columns=['stdev'])
    print('Done calculating stdev estimate.')

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
    model_name_in_paper (str) : '1_region_LP', '6_region_LP'
        or '6_region_MILP'
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
    print('Calculating point estimate...')
    point_estimate = run_years_simulation(
        model_name_in_paper=model_name_in_paper,
        startyear=point_estimate_range[0],
        endyear=point_estimate_range[1]
    )
    point_estimate = pd.DataFrame(point_estimate.values,
                                  columns=['point_estimate'],
                                  index=point_estimate.index)
    print('Done calculating point_estimate.')


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
