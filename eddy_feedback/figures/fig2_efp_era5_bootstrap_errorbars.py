"""
Uncertainty of the eddy-feedback in ERA5 estimated by bootstrapping

Show box-and-whisker plots for the range of eddy-feedback parameters calculated from
ERA5 by sampling with replacement. Show different time periods to emphasise the
difference between the periods 1940-1979 and 1979-2022
"""

import iris
from iris.analysis import MEAN
import matplotlib.pyplot as plt

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

from eddy_feedback import datadir, plotdir, bootstrapping


def main():
    data_path = datadir / "eddy_feedback/daily_mean"

    n_samples = 1000
    months = ["Dec", "Jan", "Feb"]
    month_cs = iris.Constraint(month=months)
    year_blocks = [(1941, 2022), (1941, 1979), (1980, 2022)]

    # Store a list of each result labelled by the year range and pressure levels
    labels = []
    efp = []
    efp_full = []

    for plevs, suffix in [("500hPa", "NDJFM"), ("600-200hPa", "600-200hPa_DJF")]:
        ep_flux = iris.load_cube(data_path / f"era5_daily_EP-flux-divergence_{suffix}.nc", month_cs)
        ep_flux = ep_flux.aggregated_by("season_year", MEAN)[1:-1]
        u_zm = iris.load_cube(data_path / f"era5_daily_zonal-mean-zonal-wind_{suffix}.nc", month_cs)
        u_zm = u_zm.aggregated_by("season_year", MEAN)[1:-1]

        # Calculate bootstrapped samples of the eddy-feedback parameter over the year ranges
        for start_year, end_year in year_blocks:
            labels.append("{}-{}\n{}".format(start_year-1, end_year, plevs))

            # Calculate the eddy-feedback parameter for the full period
            time_cs = iris.Constraint(
                season_year=lambda cell: start_year <= cell <= end_year
            )
            ep_flux_years = ep_flux.extract(time_cs)
            u_zm_years = u_zm.extract(time_cs)

            result_full = eddy_feedback_parameter(ep_flux_years, u_zm_years)
            if plevs == "600-200hPa":
                result_full = result_full.collapsed("pressure_level", MEAN)
            efp_full.append(result_full.data)

            # Calculate the eddy-feedback parameter over samples from the specified
            # period
            efp.append(bootstrapping.bootstrap_eddy_feedback_parameter(
                start_year=start_year,
                end_year=end_year,
                n_samples=n_samples,
                plevs=plevs,
            ))

    # Plot box-and-whisker plot for each time period
    plt.figure(figsize=(8, 5))
    positions = [1, 2, 3, 5, 6, 7]
    plt.boxplot(efp, positions=positions, whis=(2.5, 97.5))
    plt.plot(positions, efp_full, "kx")

    plt.xticks(positions, labels)
    plt.xlabel("Sampling Period")
    plt.ylabel("Eddy-Feedback Parameter")
    plt.title("Bootstrap {} samples".format(n_samples))

    plt.savefig(
        plotdir /
        f"fig2_eddy-feedback-parameter_era5_plevs_bootstrap_{n_samples}_samples.png"
    )
    plt.show()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
