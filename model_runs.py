"""Power system model runs, using Calliope framework."""


import time
import numpy as np
import pandas as pd
import samplers
import calliope


def run_model_1region(model, save_csv=False):
    """Run Calliope 1region model and return solution.

    Parameters:
    -----------
    model: Calliope model to be evaluated
    data: pandas DataFrame with demand and wind data
    save_csv: save results in custom directory name or not

    Returns:
    --------
    out: pandas DataFrame with relevant model outputs
    """

    # Run the model
    model.run(force_rerun=True)
    if save_csv is not False:
        model.to_csv(save_csv)
    R = model.results    # For concise code
    num_ts = len(model.inputs.timesteps)

    # This will be the output DataFrame
    out = pd.DataFrame(columns=['output'])

    # Insert optimal capacities
    out.loc['cap_tot_bl'] = float(R.energy_cap.loc['region1::baseload'])
    out.loc['cap_tot_pk'] = float(R.energy_cap.loc['region1::peaking'])
    out.loc['cap_tot_wd'] = float(R.resource_area.loc['region1::wind'])

    # Insert annualised generation levels
    out.loc['gen_tot_bl'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region1::baseload::power'].sum())
    out.loc['gen_tot_pk'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region1::peaking::power'].sum())
    out.loc['gen_tot_wd'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region1::wind::power'].sum())
    out.loc['gen_tot_um'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region1::unmet::power'].sum())
    out.loc['dem_tot'] = -(8760/num_ts) * \
        float(R.carrier_con.loc['region1::demand_power::power'].sum())

    # Conduct various tests to check for bugs and mistakes
    # First calculate costs by hand
    cost_bl_ins1 = 300 * out.loc['cap_tot_bl']
    cost_pk_ins1 = 100 * out.loc['cap_tot_pk']
    cost_wd_ins1 = 100 * out.loc['cap_tot_wd']
    cost_bl_gen1 = 0.005 * out.loc['gen_tot_bl']
    cost_pk_gen1 = 0.035 * out.loc['gen_tot_pk']
    cost_um_gen1 = 6 * out.loc['gen_tot_um']
    cost_tot1 = cost_bl_ins1 + cost_pk_ins1 + cost_wd_ins1 + \
        cost_bl_gen1 + cost_pk_gen1 + cost_um_gen1
    out.loc['cost'] = cost_tot1

    # Calculate costs from model outputs
    cost_bl_ins2 = (8760/num_ts) * \
        float(R.cost_investment[0].loc['region1::baseload'])
    cost_pk_ins2 = (8760/num_ts) * \
        float(R.cost_investment[0].loc['region1::peaking'])
    cost_wd_ins2 = (8760/num_ts) * \
        float(R.cost_investment[0].loc['region1::wind'])
    cost_bl_gen2 = (8760/num_ts) * \
        float(R.cost_var[0].loc['region1::baseload'].sum())
    cost_pk_gen2 = (8760/num_ts) * \
        float(R.cost_var[0].loc['region1::peaking'].sum())
    cost_um_gen2 = (8760/num_ts) * \
        float(R.cost_var[0].loc['region1::unmet'].sum())
    cost_tot2 = (8760/num_ts) * float(R.cost.sum())
    gen_tot = float(out.loc['gen_tot_bl'] + out.loc['gen_tot_pk'] +
                    out.loc['gen_tot_wd'] + out.loc['gen_tot_um'])
    dem_tot = float(out.loc['dem_tot'])

    # Conduct tests and print warnings (for diagnostics)
    # Costs
    names = ['baseload install', 'peaking install', 'wind install',
             'baseload generation', 'peaking generation',
             'unmet generation', 'total system']
    costs1 = [cost_bl_ins1, cost_pk_ins1, cost_wd_ins1, cost_bl_gen1,
              cost_pk_gen1, cost_um_gen1, cost_tot1]
    costs2 = [cost_bl_ins2, cost_pk_ins2, cost_wd_ins2, cost_bl_gen2,
              cost_pk_gen2, cost_um_gen2, cost_tot2]
    for name, cost1, cost2 in zip(names, costs1, costs2):
        if abs(float(cost1) - float(cost2)) > 10:
            print('WARNING: ' + name + ' costs do not match!')
            print('manual: %.3f, model: %.3f' %(cost1, cost2))
    # Demand balance
    if abs(float(gen_tot) - float(dem_tot)) > 0.1:
        print('WARNING: demand and supply do not match!')
        print('demand: %.3f, generation: %.3f' %(dem_tot, gen_tot))

    out.loc['emissions'] = (200*out.loc['gen_tot_bl'] + \
                            400*out.loc['gen_tot_pk'])

    return out


def run_model_6regions(model, save_csv=False):
    """Run Calliope 6region model and return solution.

    Parameters:
    -----------
    model: Calliope model to be evaluated
    save_csv: save results in custom directory name or not

    Returns:
    --------
    out: pandas DataFrame with relevant model outputs
    """

    # Run the model
    model.run(force_rerun=True)
    if save_csv is not False:
        model.to_csv(save_csv)
    R = model.results    # For concise code
    num_ts = len(model.inputs.timesteps)

    # This will be the output DataFrame
    out = pd.DataFrame(columns=['output'])

    # Insert optimal capacities
    out.loc['cap_r1_bl'] = float(R.energy_cap.loc['region1::baseload'])
    out.loc['cap_r1_pk'] = float(R.energy_cap.loc['region1::peaking'])
    out.loc['cap_r2_wd'] = float(R.resource_area.loc['region2::wind'])
    out.loc['cap_r3_bl'] = float(R.energy_cap.loc['region3::baseload'])
    out.loc['cap_r3_pk'] = float(R.energy_cap.loc['region3::peaking'])
    out.loc['cap_r5_wd'] = float(R.resource_area.loc['region5::wind'])
    out.loc['cap_r6_bl'] = float(R.energy_cap.loc['region6::baseload'])
    out.loc['cap_r6_pk'] = float(R.energy_cap.loc['region6::peaking'])
    out.loc['cap_r6_wd'] = float(R.resource_area.loc['region6::wind'])
    out.loc['cap_tr12'] = \
        float(R.energy_cap.loc['region1::transmission_other:region2'])
    out.loc['cap_tr15'] = \
        float(R.energy_cap.loc['region1::transmission_region1to5:region5'])
    out.loc['cap_tr16'] = \
        float(R.energy_cap.loc['region1::transmission_other:region6'])
    out.loc['cap_tr23'] = \
        float(R.energy_cap.loc['region2::transmission_other:region3'])
    out.loc['cap_tr34'] = \
        float(R.energy_cap.loc['region3::transmission_other:region4'])
    out.loc['cap_tr45'] = \
        float(R.energy_cap.loc['region4::transmission_other:region5'])
    out.loc['cap_tr56'] = \
        float(R.energy_cap.loc['region5::transmission_other:region6'])
    out.loc['cap_tot_bl'] = \
        out.loc[['cap_r1_bl', 'cap_r3_bl', 'cap_r6_bl']].sum()
    out.loc['cap_tot_pk'] = \
        out.loc[['cap_r1_pk', 'cap_r3_pk', 'cap_r6_pk']].sum()
    out.loc['cap_tot_wd'] = \
        out.loc[['cap_r2_wd', 'cap_r5_wd', 'cap_r6_wd']].sum()
    out.loc['cap_tot_tr_100'] = \
        out.loc[['cap_tr12', 'cap_tr16', 'cap_tr23',
                 'cap_tr34', 'cap_tr45', 'cap_tr56']].sum()
    out.loc['cap_tot_tr_150'] = out.loc['cap_tr15']

    # Insert annualised generation levels
    out.loc['gen_r1_bl'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region1::baseload::power'].sum())
    out.loc['gen_r1_pk'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region1::peaking::power'].sum())
    out.loc['gen_r2_wd'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region2::wind::power'].sum())
    out.loc['gen_r2_um'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region2::unmet::power'].sum())
    out.loc['gen_r3_bl'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region3::baseload::power'].sum())
    out.loc['gen_r3_pk'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region3::peaking::power'].sum())
    out.loc['gen_r4_um'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region4::unmet::power'].sum())
    out.loc['gen_r5_wd'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region5::wind::power'].sum())
    out.loc['gen_r5_um'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region5::unmet::power'].sum())
    out.loc['gen_r6_bl'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region6::baseload::power'].sum())
    out.loc['gen_r6_pk'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region6::peaking::power'].sum())
    out.loc['gen_r6_wd'] = (8760/num_ts) * \
        float(R.carrier_prod.loc['region6::wind::power'].sum())
    out.loc['dem_r2'] = -(8760/num_ts) * \
        float(R.carrier_con.loc['region2::demand_power::power'].sum())
    out.loc['dem_r4'] = -(8760/num_ts) * \
        float(R.carrier_con.loc['region4::demand_power::power'].sum())
    out.loc['dem_r5'] = -(8760/num_ts) * \
        float(R.carrier_con.loc['region5::demand_power::power'].sum())
    out.loc['gen_tot_bl'] = \
        out.loc[['gen_r1_bl', 'gen_r3_bl', 'gen_r6_bl']].sum()
    out.loc['gen_tot_pk'] = \
        out.loc[['gen_r1_pk', 'gen_r3_pk', 'gen_r6_pk']].sum()
    out.loc['gen_tot_wd'] = \
        out.loc[['gen_r2_wd', 'gen_r5_wd', 'gen_r6_wd']].sum()
    out.loc['gen_tot_um'] = \
        out.loc[['gen_r2_um', 'gen_r4_um', 'gen_r5_um']].sum()
    out.loc['dem_tot'] = \
        out.loc[['dem_r2', 'dem_r4', 'dem_r5']].sum()


    # Conduct various tests to check for bugs and mistakes
    # First calculate costs by hand
    cost_bl_ins1 = 300 * out.loc['cap_tot_bl']
    cost_pk_ins1 = 100 * out.loc['cap_tot_pk']
    cost_wd_ins1 = 100 * out.loc['cap_tot_wd']
    cost_tr_ins1 = 100 * out.loc['cap_tot_tr_100'] + \
        150 * out.loc['cap_tot_tr_150']
    cost_bl_gen1 = 0.005 * out.loc['gen_tot_bl']
    cost_pk_gen1 = 0.035 * out.loc['gen_tot_pk']
    cost_um_gen1 = 6 * out.loc['gen_tot_um']
    cost_tot1 = cost_bl_ins1 + cost_pk_ins1 + cost_wd_ins1 + \
        cost_tr_ins1 + cost_bl_gen1 + cost_pk_gen1 + cost_um_gen1
    out.loc['cost'] = cost_tot1

    # Calculate costs from model outputs
    cost_bl_ins2 = (8760/num_ts) * \
        float(R.cost_investment[0].loc['region1::baseload'] +
              R.cost_investment[0].loc['region3::baseload'] +
              R.cost_investment[0].loc['region6::baseload'])
    cost_pk_ins2 = (8760/num_ts) * \
        float(R.cost_investment[0].loc['region1::peaking'] +
              R.cost_investment[0].loc['region3::peaking'] +
              R.cost_investment[0].loc['region6::peaking'])
    cost_wd_ins2 = (8760/num_ts) * \
        float(R.cost_investment[0].loc['region2::wind'] +
              R.cost_investment[0].loc['region5::wind'] +
              R.cost_investment[0].loc['region6::wind'])
    cost_tr_ins2 = 2 * (8760/num_ts) * \
        float(R.cost_investment[0].loc['region1::transmission_other:region2'] +
              R.cost_investment[0].loc['region1::transmission_region1to5:region5'] +
              R.cost_investment[0].loc['region1::transmission_other:region6'] +
              R.cost_investment[0].loc['region2::transmission_other:region3'] +
              R.cost_investment[0].loc['region3::transmission_other:region4'] +
              R.cost_investment[0].loc['region4::transmission_other:region5'] +
              R.cost_investment[0].loc['region5::transmission_other:region6'])
    cost_bl_gen2 = (8760/num_ts) * \
        float(R.cost_var[0].loc['region1::baseload'].sum() +
              R.cost_var[0].loc['region3::baseload'].sum() +
              R.cost_var[0].loc['region6::baseload'].sum())
    cost_pk_gen2 = (8760/num_ts) * \
        float(R.cost_var[0].loc['region1::peaking'].sum() +
              R.cost_var[0].loc['region3::peaking'].sum() +
              R.cost_var[0].loc['region6::peaking'].sum())
    cost_um_gen2 = (8760/num_ts) * \
        float(R.cost_var[0].loc['region2::unmet'].sum() +
              R.cost_var[0].loc['region4::unmet'].sum() +
              R.cost_var[0].loc['region5::unmet'].sum())
    cost_tot2 = (8760/num_ts) * float(R.cost.sum())
    gen_tot = float(out.loc['gen_tot_bl'] + out.loc['gen_tot_pk'] +
                    out.loc['gen_tot_wd'] + out.loc['gen_tot_um'])
    dem_tot = float(out.loc['dem_tot'])

    # Conduct tests and print warnings (for diagnostics)
    # Costs
    names = ['baseload install', 'peaking install', 'wind install',
             'transmission install', 'baseload generation',
             'peaking generation', 'unmet generation', 'total system']
    costs1 = [cost_bl_ins1, cost_pk_ins1, cost_wd_ins1, cost_tr_ins1,
              cost_bl_gen1, cost_pk_gen1, cost_um_gen1, cost_tot1]
    costs2 = [cost_bl_ins2, cost_pk_ins2, cost_wd_ins2, cost_tr_ins2,
              cost_bl_gen2, cost_pk_gen2, cost_um_gen2, cost_tot2]
    for name, cost1, cost2 in zip(names, costs1, costs2):
        if abs(float(cost1) - float(cost2)) > 10:
            print('WARNING: ' + name + ' costs do not match!')
            print('manual: %.3f, model: %.3f' %(cost1, cost2))
    # Demand balance
    if abs(float(gen_tot) - float(dem_tot)) > 0.1:
        print('WARNING: demand and supply do not match!')
        print('demand: %.3f, generation: %.3f' %(dem_tot, gen_tot))

    out.loc['emissions'] = (200*out.loc['gen_tot_bl'] + \
                            400*out.loc['gen_tot_pk'])

    return out


def run_simulation(model_name, data, save_csv=False):
    """Run Calliope model with some demand & wind data.

    Parameters:
    -----------
    model_name: e.g. '1region_cont' or '6regions_disc'
    data: demand & wind data (should match the model, e.g. 1 region models
        have demand and wind as columns, whereas 6 region models have
        demand and wind in the correct regions.
    save_csv: whether to save larger set of outputs (the full Callope
        set of outputs) as separate CSV files (insert directory name)

    Returns:
    --------
    results: pandas DataFrame with model outputs
    """

    start = time.time()

    # Create the model with possible time subset. Calliope requires a
    # CSV file to be present when initialising the model, so it uses
    # the 'demand_wind.csv' files (with all zeros) in the model directory
    last_ts = str(pd.date_range(start='1980-01-01 00:00:00',
                                freq='h', periods=data.shape[0])[-1])
    override_dict = {'model.subset_time':
                     ['1980-01-01 00:00:00', last_ts]}
    model = calliope.Model('models/' + model_name + '/model.yaml',
                           override_dict=override_dict)

    # Load on correct time series data
    if '1region' in model_name:
        tseries_in = model.inputs.resource
        tseries_in.loc['region1::demand_power'].values[:] = \
            -data.loc[:, 'demand']
        tseries_in.loc['region1::wind'].values[:] = \
            data.loc[:, 'wind']

    # Load on correct time series data
    if '6regions' in model_name:
        # results = run_model_6regions(model, save_csv=save_csv)
        tseries_in = model.inputs.resource
        tseries_in.loc['region2::demand_power'].values[:] = \
            -data.loc[:, 'demand_region2']
        tseries_in.loc['region4::demand_power'].values[:] = \
            -data.loc[:, 'demand_region4']
        tseries_in.loc['region5::demand_power'].values[:] = \
            -data.loc[:, 'demand_region5']
        tseries_in.loc['region2::wind'].values[:] = \
            data.loc[:, 'wind_region2']
        tseries_in.loc['region5::wind'].values[:] = \
            data.loc[:, 'wind_region5']
        tseries_in.loc['region6::wind'].values[:] = \
            data.loc[:, 'wind_region6']

    # Run model
    if '1region' in model_name:
        results = run_model_1region(model, save_csv=save_csv)
    if '6regions' in model_name:
        results = run_model_6regions(model, save_csv=save_csv)

    finish = time.time()
    results.loc['time'] = finish - start

    return results


def run_years_simulation(model_name, startyear, endyear, save_csv=False):
    """Run Calliope model with certain years of data.

    Parameters:
    -----------
    model_name: e.g. '1region_cont' or '6regions_disc'
    startyear: first year of data to run model on
    lastyear: last year of data to run model on
    save_csv: whether to save csv outputs (insert directory name)

    Returns:
    --------
    results: pandas DataFrame with model outputs
    """

    # Import correct data
    if '1region' in model_name:
        data = pd.read_csv('data/demand_wind_1region.csv', index_col=0)
    if '6regions' in model_name:
        data = pd.read_csv('data/demand_wind_6regions.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.loc[str(startyear):str(endyear)]

    # Run model
    results = run_simulation(model_name, data=data, save_csv=save_csv)

    return results


def run_bootstrap_simulation(model_name, scheme, num_blocks_per_bin,
                             save_csv=False):
    """Run model with bootstrap sampled data

    Parameters:
    -----------
    model_name e.g. '1region_cont' or '6regions_disc'
    scheme: either 'months' or 'weeks' -- scheme used to create bootstrap
        samples
    num_blocks_per_bin: either the number of months sampled from each
        calendar month, or the number of weeks sampled from each season
    save_csv: whether to save csv outputs (insert directory name)

    Returns:
    --------
    results: pandas DataFrame with model outputs
    """

    # Import correct data
    if '1region' in model_name:
        data = pd.read_csv('data/demand_wind_1region.csv', index_col=0)
    if '6regions' in model_name:
        data = pd.read_csv('data/demand_wind_6regions.csv', index_col=0)
    data.index = pd.to_datetime(data.index)

    # Create bootstrap sample and run model
    if scheme == 'months':
        sample = samplers.bootstrap_sample_months(data, num_blocks_per_bin)
    elif scheme == 'weeks':
        sample = samplers.bootstrap_sample_weeks(data, num_blocks_per_bin)
    else:
        raise ValueError('Must be either months or weeks scheme')

    # Run the model
    results = run_simulation(model_name, data=sample, save_csv=save_csv)

    return results


def run_buq_algorithm(model_name,
                      point_sample_length,
                      bootstrap_scheme,
                      num_blocks_per_bin,
                      num_bootstrap_samples):
    """Run through BUQ algorithm once to estimate standard deviation.

    Parameters:
    -----------
    model_name (str) : e.g. '1region_cont' or '6regions_disc'
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
    if bootstrap_scheme == 'months':
        bootstrap_sample_length = num_blocks_per_bin * 8760

    # Calculate variance across bootstrap samples
    print('Calculating stdev estimate...')
    print('Starting bootstrap samples')

    # Run model for each bootstrap sample
    run_index = np.arange(num_bootstrap_samples)
    for sample_num in run_index:
        print('Calculating bootstrap sample {}...'.format(sample_num+1))
        results = run_bootstrap_simulation(model_name,
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
    point_estimate_variance = \
        (bootstrap_sample_length/point_sample_length) * bootstrap_variance
    point_estimate_stdev = pd.DataFrame(np.sqrt(point_estimate_variance),
                                        columns=['stdev'])
    print('Done calculating stdev estimate.')

    return point_estimate_stdev


def calculate_point_estimate_and_stdev(model_name,
                                       point_estimate_range,
                                       bootstrap_scheme,
                                       num_blocks_per_bin,
                                       num_bootstrap_samples):
    """Calculate point estimate using a single long simulation and estimate
    standard deviation using multiple short simulations and BUQ algorithm.

    Parameters:
    -----------
    model_name (str) : e.g. '1region_cont' or '6regions_disc'
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
    point_estimate = run_years_simulation(model_name=model_name,
                                          startyear=point_estimate_range[0],
                                          endyear=point_estimate_range[1])
    point_estimate = pd.DataFrame(point_estimate.values,
                                  columns=['point_estimate'],
                                  index=point_estimate.index)
    print('Done calculating point_estimate.')


    # Estimate standard deviation with BUQ algorithm
    point_estimate_stdev = \
        run_buq_algorithm(model_name=model_name,
                          point_sample_length=point_sample_length,
                          bootstrap_scheme=bootstrap_scheme,
                          num_blocks_per_bin=num_blocks_per_bin,
                          num_bootstrap_samples=num_bootstrap_samples)
    point_estimate_stdev = pd.DataFrame(point_estimate_stdev.values,
                                        columns=['stdev'],
                                        index=point_estimate_stdev.index)

    # Create single dataframe with point and standard deviation estimate
    estimate_with_stdev = point_estimate.join(point_estimate_stdev)

    return estimate_with_stdev
