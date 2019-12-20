import numpy as np
import pandas as pd
import pdb


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
            sample_index = 24*np.random.choice(possible_startdays) + \
                np.arange(7*24)
            sample = data_sel.iloc[sample_index]
            output[k:k+sample.shape[0]] = sample.values
            k = k + sample.shape[0]

    # Change output from numpy array to pandas DataFrame
    if data.shape[1] == 2:
        output_columns = ['demand', 'wind']
    if data.shape[1] == 6:
        output_columns = \
            ['demand_region2', 'demand_region4', 'demand_region5',
             'wind_region2', 'wind_region5', 'wind_region6']
    output = pd.DataFrame(output, index=np.arange(sample_length),
                          columns=output_columns)

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
        output_columns = \
            ['demand_region2', 'demand_region4', 'demand_region5',
             'wind_region2', 'wind_region5', 'wind_region6']
    output = pd.DataFrame(years_np, index=np.arange(years_np.shape[0]),
                          columns=output_columns)

    return output


# def bootstrap_sample_periods(data, block_length, num_blocks_per_bin,
#                              bins=[[i] for i in np.arange(12)+1]):
#     """Create bootstrap sample from time series

#     Parameters:
#     -----------
#     data: pandas DataFrame with demand and wind data
#     block_length: block length (in days)
#     num_blocks_per_bin: number of blocks chosen from each bin (e.g.
#         number of blocks from each month)
#     bins: list of lists with the months in each bin. For example, if each
#         bin is two months, bins=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
#     """

#     num_bins = len(bins)
#     sample_length = num_bins*num_blocks_per_bin*block_length*24
#     output = np.zeros(shape=(sample_length, data.shape[1]))
#     k = 0
#     for block in range(num_blocks_per_bin):
#         for bin_num in range(num_bins):
#             year = np.random.choice(list(data.index.year.unique()))
#             data_sel = data[(data.index.year == year)&
#                             (data.index.month.isin(bins[bin_num]))]
#             num_days = data_sel.shape[0]/24
#             possible_startdays = \
#                 np.arange(num_days - block_length + 1)
#             sample_index = 24*np.random.choice(possible_startdays) + \
#                 np.arange(block_length*24)
#             sample = data_sel.iloc[sample_index]
#             # print(sample)
#             output[k:k+sample.shape[0]] = sample.values
#             k = k + sample.shape[0]

#     # Change output from numpy array to pandas DataFrame
#     if data.shape[1] == 2:
#         output_columns = ['demand', 'wind']
#     if data.shape[1] == 6:
#         output_columns = \
#             ['demand_region2', 'demand_region4', 'demand_region5',
#              'wind_region2', 'wind_region5', 'wind_region6']
#     output = pd.DataFrame(output, index=np.arange(sample_length),
#                           columns=output_columns)

#     return output


# def bootstrap_months(data, num_years):
#     """"Create hypothetical years by block bootstrapping months.

#     Parameters:
#     -----------
#     data: dataset with demand and wind data
#     num_years: number of years of the output sample

#     Returns:
#     --------
#     year_np: numpy array, shape=(8760, 6), with the demand and
#         wind values across  hypothetical year
#     """

#     data_np = np.array(data)
#     years_np = np.zeros(shape=(8760*num_years, data.shape[1]))
#     num_years_inp = data_np.shape[0]/8760

#     # Create each year individually and input them
#     for year_num in range(num_years):
#         year_np = np.zeros(shape=(8760, data.shape[1]))
#         lims = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832,
#                 6552, 7296, 8016, 8760]
#         # List of years from which months are taken
#         month_years = np.array([int(num_years_inp*np.random.rand(1))
#                                 for month in range(12)])
#         # Input the sampled months
#         for month in range(12):
#             llim, rlim = lims[month], lims[month+1]
#             yrstart = 8760 * month_years[month]
#             year_np[llim:rlim] = data_np[yrstart+llim:yrstart+rlim]

#         # Input the year into the years array
#         years_np[8760*year_num:8760*(year_num+1)] = year_np

#     # Change output from numpy array to pandas DataFrame
#     if data.shape[1] == 2:
#         output_columns = ['demand', 'wind']
#     if data.shape[1] == 6:
#         output_columns = \
#             ['demand_region2', 'demand_region4', 'demand_region5',
#              'wind_region2', 'wind_region5', 'wind_region6']
#     output = pd.DataFrame(years_np, index=np.arange(years_np.shape[0]),
#                           columns=output_columns)

#     return output


def determine_statistics(directory, sample_list):
    """Determine mean and variance of outputs across samples.

    Parameters:
    -----------
    directory: directory in which model outputs are stored
    sample_list: name of output files (e.g. ['bs_7d_1b', 'bs_7d_2b',
        'years'])

    Returns:
    --------
    mean: pandas DataFrame with mean values across outputs
    var: pandas DataFrame with variance values across outputs
    """

    # Create DataFrame and input relevant results
    columns = pd.read_csv(directory + 'years.csv', index_col=0).columns
    mean = pd.DataFrame(index=sample_list, columns=columns)
    var = pd.DataFrame(index=sample_list, columns=columns)
    for sample in sample_list:
        outputs = pd.read_csv(directory + sample + '.csv',
                              index_col=[0])
        mean.loc[sample] = np.mean(outputs, axis=0)
        var.loc[sample] = np.var(outputs, axis=0)

    return mean, var


def variance_estimates(sample_length):
    """Find extrapolation estimates of variances from model runs of length
    1 year.

    Parameters:
    -----------
    sample_length: number of runs of length 1y variance is taken across

    Returns:
    --------
    vars_estimates: variance estimates across samples of length sample_length
    """

    directory = 'outputs/1region/3techs/current/'

    outputs_1y = pd.read_csv(directory + '1y.csv', index_col=0)
    vars_estimates = pd.DataFrame(np.zeros(shape=(5, 9)), index=np.arange(5),
                                  columns=outputs_1y.columns)

    for sample in range(1000):
        indices = np.random.choice(np.arange(1000), sample_length,
                                   replace=True)
        outputs_sample = outputs_1y.loc[indices]
        vars_estimates.loc[sample] = \
            np.var(outputs_sample, axis=0, ddof=1)/10    # ddof=1, unbiased

    vars_estimates.to_csv(directory + 'vars_estimates_' +
                          str(sample_length) + 'samples.csv',
                          float_format='%.4g')

    # return vars_estimates


def calc_gens(data, caps):
    """Calculate generation across technologies and timesteps."""

    gens = pd.DataFrame(index=data.index,
                        columns=['baseload', 'peaking', 'wind', 'unmet'])
    cap_bl, cap_pk, cap_wd = caps

    net_demand = data['demand'] - cap_wd * data['wind']
    net_demand[net_demand < 0] = 0

    gens.loc[:, 'baseload'] = np.minimum(net_demand, cap_bl)
    gens.loc[:, 'peaking'] = \
        np.minimum(net_demand - gens['baseload'], cap_pk)
    gens.loc[:, 'wind'] = data['demand'] - net_demand
    gens.loc[:, 'unmet'] = net_demand - gens['baseload'] - gens['peaking']

    return gens
