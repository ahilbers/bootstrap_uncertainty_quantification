"""Bootstrap sampling algorithms: the 'months' and 'weeks' schemes."""


import numpy as np
import pandas as pd


def bootstrap_sample_weeks(data, num_weeks_per_season):
    """Create bootstrap sample by sampling weeks from different
    meteorological seasons.

    Parameters:
    -----------
    data: pandas DataFrame with demand and wind data
    num_weeks_per_season: number of weeks sampled from each season
    """

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
    data: dataset with demand and wind data
    num_years: number of years of the output sample

    Returns:
    --------
    year_np: numpy array, shape=(8760, 6), with the demand and
        wind values across  hypothetical year
    """

    data_np = np.array(data)
    years_np = np.zeros(shape=(8760*num_years, data.shape[1]))
    num_years_inp = data_np.shape[0]/8760

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
            year_np[llim:rlim] = data_np[yrstart+llim:yrstart+rlim]

        # Input the year into the years array
        years_np[8760*year_num:8760*(year_num+1)] = year_np

    # Change output from numpy array to pandas DataFrame
    if data.shape[1] == 2:
        output_columns = ['demand', 'wind']
    if data.shape[1] == 6:
        output_columns = ['demand_region2', 'demand_region4',
                          'demand_region5', 'wind_region2',
                          'wind_region5', 'wind_region6']
    index = pd.to_datetime(np.arange(years_np.shape[0]),
                           origin='2020', unit='h')  # Dummy datetime index
    output = pd.DataFrame(years_np, index=index, columns=output_columns)

    return output
