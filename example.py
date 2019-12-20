import os
import scripts as sc
import calliope


def run_example():
    """Run an example of the methodology. 
    """

    # Arguments: change these as desired
    model_name = '1region_cont'
    point_estimate_range = [2017, 2017]
    bootstrap_scheme = 'weeks'
    num_blocks_per_bin = 4
    num_bootstrap_samples = 10

    # Run the methodology, return point estimates and stdev estimates
    results = sc.estimate_point_and_stdev(
        model_name=model_name,
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
