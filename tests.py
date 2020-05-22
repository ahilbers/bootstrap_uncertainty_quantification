"""Tests to check whether the models behave as required."""


import logging
import numpy as np
import pandas as pd
import buq


# Install costs and generation costs. These should match the information
# provided in the model.yaml and techs.yaml files in the model definition
COSTS = pd.DataFrame(columns=['install', 'generation'])

# Heterogenised costs for 6 region model
COSTS.loc['ccgt_region1']    = [100.1, 0.035001]
COSTS.loc['ocgt_region1']    = [ 50.1, 0.100001]
COSTS.loc['wind_region2']    = [ 90.2, 0.000002]
COSTS.loc['nuclear_region3'] = [300.3, 0.005003]
COSTS.loc['ccgt_region3']    = [100.3, 0.035003]
COSTS.loc['wind_region5']    = [100.5, 0.000005]
COSTS.loc['ocgt_region6']    = [ 50.6, 0.100006]
COSTS.loc['wind_region6']    = [ 70.6, 0.000006]
COSTS.loc['unmet_region2']   = [    0, 6.000002]
COSTS.loc['unmet_region4']   = [    0, 6.000004]
COSTS.loc['unmet_region5']   = [    0, 6.000005]
COSTS.loc['transmission_region1_region2'] = [100.12, 0]
COSTS.loc['transmission_region1_region5'] = [150.15, 0]
COSTS.loc['transmission_region1_region6'] = [130.16, 0]
COSTS.loc['transmission_region2_region3'] = [100.23, 0]
COSTS.loc['transmission_region3_region4'] = [100.34, 0]
COSTS.loc['transmission_region4_region5'] = [100.45, 0]
COSTS.loc['transmission_region5_region6'] = [100.56, 0]


# Topology of 6 region model. These should match the information provided
# in the locations.yaml files in the model definition
NUCLEAR_TOP, CCGT_TOP, OCGT_TOP, WIND_TOP, UNMET_TOP, DEMAND_TOP = (
    [('nuclear', i) for i in ['region3']],
    [('ccgt', i) for i in ['region1', 'region3']],
    [('ocgt', i) for i in ['region1', 'region6']],
    [('wind', i) for i in ['region2', 'region5', 'region6']],
    [('unmet', i) for i in ['region2', 'region4', 'region5']],
    [('demand', i) for i in ['region2', 'region4', 'region5']]
)
TRANSMISSION_TOP = [('transmission', *i)
                    for i in [('region1', 'region2'),
                              ('region1', 'region5'),
                              ('region1', 'region6'),
                              ('region2', 'region3'),
                              ('region3', 'region4'),
                              ('region4', 'region5'),
                              ('region5', 'region6')]]


def test_output_consistency_6_region(model, run_mode):
    """Check if model outputs are internally consistent for 6 region model.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    run_mode (str) : 'plan' or 'operate'

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    passing = True
    cost_total_method1 = 0

    out = model.get_summary_outputs()
    res = model.results
    corrfac = 8760/model.num_timesteps    # For annualisation

    # Test if generation technology installation costs are consistent
    if run_mode == 'plan':
        for tech, region in NUCLEAR_TOP + CCGT_TOP + OCGT_TOP + WIND_TOP:
            cost_method1 = float(
                COSTS.loc['{}_{}'.format(tech, region), 'install'] *
                out.loc['cap_{}_{}'.format(tech, region)]
            )
            cost_method2 = corrfac * float(
                res.cost_investment[0].loc['{}::{}_{}'.format(region,
                                                              tech,
                                                              region)]
            )
            if abs(cost_method1 - cost_method2) > 0.1:
                logging.error('FAIL: %s install costs in %s do not match!\n'
                              '    manual: %s, model: %s',
                              tech, region, cost_method1, cost_method2)
                passing = False
            cost_total_method1 += cost_method1

    # Test if transmission installation costs are consistent
    if run_mode == 'plan':
        for tech, region_a, region_b in TRANSMISSION_TOP:
            cost_method1 = float(
                COSTS.loc[
                    '{}_{}_{}'.format(tech, region_a, region_b), 'install'
                ] * out.loc[
                    'cap_transmission_{}_{}'.format(region_a, region_b)
                ]
            )
            cost_method2 = 2 * corrfac * \
                float(res.cost_investment[0].loc[
                    '{}::{}_{}_{}:{}'.format(region_a,
                                             tech,
                                             region_a,
                                             region_b,
                                             region_b)
                ])
            if abs(cost_method1 - cost_method2) > 0.1:
                logging.error('FAIL: %s install costs from %s to %s do '
                              'not match!\n    manual: %s, model: %s',
                              tech, region_a, region_b,
                              cost_method1, cost_method2)
                passing = False
            cost_total_method1 += cost_method1

    # Test if generation costs are consistent
    for tech, region in NUCLEAR_TOP + CCGT_TOP + OCGT_TOP + WIND_TOP + UNMET_TOP:
        cost_method1 = float(
            COSTS.loc['{}_{}'.format(tech, region), 'generation']
            * out.loc['gen_{}_{}'.format(tech, region)]
        )
        cost_method2 = corrfac * float(
            res.cost_var[0].loc['{}::{}_{}'.format(region, tech, region)].sum()
        )
        if abs(cost_method1 - cost_method2) > 0.1:
            logging.error('FAIL: %s generation costs in %s do not match!\n'
                          '    manual: %s, model: %s',
                          tech, region, cost_method1, cost_method2)
            passing = False
        cost_total_method1 += cost_method1

    # Test if total costs are consistent
    if run_mode == 'plan':
        cost_total_method2 = corrfac * float(res.cost.sum())
        if abs(cost_total_method1 - cost_total_method2) > 0.1:
            logging.error('FAIL: total system costs do not match!\n'
                          '    manual: %s, model: %s',
                          cost_total_method1, cost_total_method2)
            passing = False

    # Test if supply matches demand
    generation_total = float(out.loc[['gen_nuclear_total',
                                      'gen_ccgt_total',
                                      'gen_ocgt_total',
                                      'gen_wind_total',
                                      'gen_unmet_total']].sum())
    demand_total = float(out.loc['demand_total'])
    if abs(generation_total - demand_total) > 0.1:
        logging.error('FAIL: generation does not match demand!\n'
                      '    generation: %s, demand: %s',
                      generation_total, demand_total)
        passing = False

    return passing


def test_outputs_against_benchmarks():
    """Test model outputs against benchmarks."""

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=getattr(logging, 'INFO'),
        datefmt='%Y-%m-%d,%H:%M:%S'
    )

    # Run each test simulation and see if results match benchmarks
    benchmarks = {'LP_planning': 'test_benchmarks/LP_planning.csv',
                  'operation': 'test_benchmarks/operation.csv'}
    passing_all_benchmarks = []
    for benchmark_name in benchmarks:
        logging.info('\n\n\n\n\nStarting bechmark comparison for %s model',
                     benchmark_name)
        passing = True
        np.random.seed(42)
        estimate_with_stdev = buq.calculate_point_estimate_and_stdev(
            model_name_in_paper=benchmark_name,
            point_estimate_range=[2017, 2017],
            bootstrap_scheme='weeks',
            num_blocks_per_bin=1,
            num_bootstrap_samples=10
        )[:-1]
        benchmark_values = pd.read_csv(benchmarks[benchmark_name],
                                       index_col=0)[:-1]
        if not np.allclose(estimate_with_stdev, benchmark_values):
            logging.error(
                'FAIL: Model outputs do not match benchmark outputs!\n'
                'Model outputs: \n%s\n \nBenchmark outputs:\n%s\n',
                estimate_with_stdev, benchmark_values
            )
            passing = False
        else:
            logging.info('Outputs match benchmark.')
        passing_all_benchmarks.append(passing)

    passing = all(passing_all_benchmarks)
    if not passing:
        logging.error('Some benchmarks have failed! '
                      'See log above for details')
    else:
        logging.info('All benchmarks have passed.')

    return passing


if __name__ == '__main__':
    test_outputs_against_benchmarks()
