"""Code for the bootstrap uncertainty quantification (BUQ) algorithm."""


import time
import logging
import numpy as np
import pandas as pd
import buq
import models
import tests


def import_time_series_data():
    """Import time series data for model, without any time slicing."""
    ts_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    return ts_data


def bootstrap_sample_weeks(data, num_weeks_per_season):
    """Create bootstrap sample by sampling weeks from different
    meteorological seasons.

    Parameters:
    -----------
    data (pandas DataFrame) : demand and wind data
    num_weeks_per_season (int) : number of weeks sampled from each season

    Returns:
    --------
    output (pandas DataFrame) : the bootstrap sample
    """

    # Sample weeks from the meteorological seasons
    bins = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    sample_length = 4*num_weeks_per_season*7*24
    output = np.zeros(shape=(sample_length, data.shape[1]))
    k = 0
    for block in range(num_weeks_per_season):
        for bin_num in range(4):
            year = np.random.choice(list(data.index.year.unique()))
            data_sel = data[(data.index.year == year)&
                            (data.index.month.isin(bins[bin_num]))]
            num_days = data_sel.shape[0]/24
            possible_startdays = np.arange(num_days - 7 + 1)
            sample_index = (24*np.random.choice(possible_startdays)
                            + np.arange(7*24))
            sample = data_sel.iloc[sample_index]
            output[k:k+sample.shape[0]] = sample.values
            k = k + sample.shape[0]

    # Change output from numpy array to pandas DataFrame
    if data.shape[1] == 2:
        output_columns = ['demand', 'wind']
    if data.shape[1] == 6:
        output_columns = ['demand_region2', 'demand_region4',
                          'demand_region5', 'wind_region2',
                          'wind_region5', 'wind_region6']
    index = pd.to_datetime(np.arange(sample_length),
                           origin='2020', unit='h')  # Dummy datetime index
    output = pd.DataFrame(output, index=index, columns=output_columns)

    return output


def bootstrap_sample_months(data, num_years):
    """"Create hypothetical years by block bootstrapping months.

    Parameters:
    -----------
    data (pandas DataFrame) : demand and wind data
    num_years (int) : number of years of the output sample

    Returns:
    --------
    output (pandas DataFrame) : the bootstrap sample
    """

    years_np = np.zeros(shape=(8760*num_years, data.shape[1]))
    num_years_inp = data.values.shape[0]/8760

    # Create each year individually and input them
    for year_num in range(num_years):
        year_np = np.zeros(shape=(8760, data.shape[1]))
        lims = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832,
                6552, 7296, 8016, 8760]
        # List of years from which months are taken
        month_years = np.array([int(num_years_inp*np.random.rand(1))
                                for month in range(12)])
        # Input the sampled months
        for month in range(12):
            llim, rlim = lims[month], lims[month+1]
            yrstart = 8760 * month_years[month]
            year_np[llim:rlim] = data.values[yrstart+llim:yrstart+rlim]

        # Input the year into the years array
        years_np[8760*year_num:8760*(year_num+1)] = year_np

    # Change output from numpy array to pandas DataFrame
    output_columns = ['demand_region2', 'demand_region4', 'demand_region5',
                      'wind_region2', 'wind_region5', 'wind_region6']
    index = pd.to_datetime(np.arange(years_np.shape[0]),
                           origin='2020', unit='h')  # Dummy datetime index
    output = pd.DataFrame(years_np, index=index, columns=output_columns)

    return output


def run_simulation(model_name_in_paper, ts_data, run_id=0):
    """Run Calliope model with demand & wind data.

    Parameters:
    -----------
    model_name_in_paper (str) : 'LP_planning', 'MILP_planning' or
        'operation'
    ts_data (pandas DataFrame) : demand & wind time series data
    run_id (int or str) : unique id, useful if running in parallel

    Returns:
    --------
    results (pandas DataFrame) : model outputs
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
    """Run model with certain years of data."""
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
        'operation'
    scheme: either 'months' or 'weeks' -- scheme used to create bootstrap
        samples
    num_blocks_per_bin: either the number of months sampled from each
        calendar month, or the number of weeks sampled from each season

    Returns:
    --------
    results (pandas DataFrame) : model outputs
    """

    ts_data = import_time_series_data()

    # Create bootstrap sample and run model
    if scheme == 'months':
        sample = buq.bootstrap_sample_months(ts_data,
                                             num_blocks_per_bin)
    elif scheme == 'weeks':
        sample = buq.bootstrap_sample_weeks(ts_data,
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
        'operation'
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
    logging.info('Starting bootstrap samples')

    # Run model for each bootstrap sample
    run_index = np.arange(num_bootstrap_samples)
    for sample_num in run_index:
        logging.info('\n\nCalculating bootstrap sample %s', sample_num+1)
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
        'operation'
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
    logging.info('Calculating stdev estimate...')
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
    logging.info('Done calculating stdev estimate.')

    # Create single dataframe with point and standard deviation estimate
    estimate_with_stdev = point_estimate.join(point_estimate_stdev)

    return estimate_with_stdev
