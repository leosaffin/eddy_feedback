"""
Uncertainty of the eddy-feedback in ERA5 estimated by bootstrapping

Show box-and-whisker plots for the range of eddy-feedback parameters calculated from
ERA5 by sampling with replacement. Show different time periods to emphasise the
difference between the periods 1940-1979 and 1979-2022
"""

from tqdm import tqdm
import iris
from iris.analysis import MEAN
import numpy as np
import matplotlib.pyplot as plt

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

from eddy_feedback import datadir, plotdir
from eddy_feedback.nao_variance import season_mean


def main():
    n_resamples = 100
    months = ["Dec", "Jan", "Feb"]
    seasons = ["ndjfma", "mjjaso"]
    year_blocks = [(1940, 2022), (1940, 1979), (1979, 2022)]

    data_path = datadir / "constrain/eddy_feedback/daily_mean"
    plt.figure(figsize=(8, 5))

    ep_flux = iris.load_cube(data_path / "era5_daily_EP-flux-divergence_NDJFM.nc")
    ep_flux = season_mean(ep_flux, months, seasons)
    u_zm = iris.load_cube(data_path / "era5_daily_zonal-mean-zonal-wind_NDJFM.nc")
    u_zm = season_mean(u_zm, months, seasons)

    # Store a list of each result labelled by the year range
    efp = dict()

    # Calculate bootstrapped samples of the eddy-feedback parameter over the year ranges
    for year_block in year_blocks:
        label = "{}-{}, 500hPa".format(*year_block)
        efp[label] = []
        cs = iris.Constraint(
            season_year=lambda cell: year_block[0] < cell <= year_block[1]
        )

        ep_flux_years = ep_flux.extract(cs)
        u_zm_years = u_zm.extract(cs)

        efp[label] = bootstrap_eddy_feedback_parameter(
            ep_flux_years, u_zm_years, n_resamples
        )

    # Repeat for data on pressure levels
    ep_flux = iris.load_cube(
        data_path / "era5_daily_EP-flux-divergence_600-200hPa_NDJFM.nc"
    )
    ep_flux = season_mean(ep_flux, months, seasons)
    ep_flux = ep_flux.collapsed("pressure_level", MEAN)
    u_zm = iris.load_cube(
        data_path / "era5_daily_zonal-mean-zonal-wind_600-200hPa_NDJFM.nc"
    )
    u_zm = season_mean(u_zm, months, seasons)
    u_zm = u_zm.collapsed("pressure_level", MEAN)

    label = "1979-2022, 600-200hPa"
    efp[label] = bootstrap_eddy_feedback_parameter(ep_flux, u_zm, n_resamples)

    # Plot box-and-whisker plot for each time period
    for n, label in enumerate(efp):
        plt.boxplot(efp[label], positions=[n], whis=(2.5, 97.5))

    plt.xticks(range(len(efp)), efp.keys())
    plt.xlabel("Sampling Period")
    plt.ylabel("Eddy-Feedback Parameter")
    plt.title("Bootstrap {} samples".format(n_resamples))

    plt.savefig(
        plotdir /
        f"eddy-feedback-parameter_era5_plevs_bootstrap_{n_resamples}_samples.png"
    )
    plt.show()


def bootstrap_eddy_feedback_parameter(ep_flux, u_zm, n_resamples):
    efp = []
    years = np.array(ep_flux.coord("season_year").points, dtype=int)
    for _ in tqdm(range(n_resamples)):
        samples = np.random.randint(0, len(years) - 1, size=len(years))
        ep_flux_sub = ep_flux[samples]
        u_zm_sub = u_zm[samples]

        efp.append(eddy_feedback_parameter(ep_flux_sub, u_zm_sub).data)

    return efp


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
