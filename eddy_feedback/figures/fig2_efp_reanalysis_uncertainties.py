"""
Uncertainty of the eddy-feedback in reanalysis

Panel (a) Estimated by bootstrapping

Show box-and-whisker plots for the range of eddy-feedback parameters calculated from
ERA5 by sampling with replacement. Show different time periods to emphasise the
difference between the periods 1940-1979 and 1979-2022

Panel (b) 23-year rolling window.
"""

from tqdm import tqdm
import iris
from iris.analysis import MEAN
from iris.coords import DimCoord
import iris.plot as iplt
import matplotlib.pyplot as plt

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

from eddy_feedback import datadir, plotdir, bootstrapping


data_path = datadir / "eddy_feedback/daily_mean"


def main():
    n_samples = 1000
    window_size = 23
    months = ["Dec", "Jan", "Feb"]

    fig, axes = plt.subplots(1, 2, sharey="all", figsize=(8, 5))

    panel_a(axes[0], n_samples, months)
    panel_b(axes[1], window_size, months)

    axes[0].set_ylim(0, 0.6)
    axes[0].text(2, 0.55, "500hPa", ha="center", bbox=dict(facecolor='none', edgecolor='black'))
    axes[0].text(6, 0.55, "200-600hPa", ha="center", bbox=dict(facecolor='none', edgecolor='black'))
    axes[0].set_xlabel("Sampling Period")
    axes[0].set_ylabel("Eddy-Feedback Parameter")
    axes[0].set_title("Sampling Uncertainty")
    axes[0].text(0.01, 1.02, "(a)", transform=axes[0].transAxes)

    axes[1].set_xlabel("Year")
    axes[1].legend()
    axes[1].set_title("Multidecadal Variability")
    axes[1].axvline(2005, color="k")
    axes[1].text(0.01, 1.02, "(b)", transform=axes[1].transAxes)

    fig.suptitle("Eddy-Feedback Parameter - Reanalysis Uncertainties")
    fig.autofmt_xdate()

    plt.savefig(plotdir / f"fig2_efp_reanalysis_uncertainties.png")
    plt.show()


def panel_a(ax, n_samples, months):
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
        for n, (start_year, end_year) in enumerate(year_blocks):
            labels.append(f"{start_year-1}-{end_year}")

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
    positions = [1, 2, 3, 5, 6, 7]
    ax.boxplot(efp, positions=positions, whis=(2.5, 97.5))
    ax.plot(positions, efp_full, "kx")
    ax.set_xticks(positions, labels)


def panel_b(ax, window_size, months):
    plt.axes(ax)
    cs = iris.Constraint(month=months)

    for n, (reanalysis, pressure_levels, months_str, linestyle) in enumerate([
        ("ERA5", "500hPa", "NDJFM", "-k"),
        ("ERA5", "600-200hPa", "DJF", "--k"),
        ("ERA20c", "500hPa", "DJF", "-.k"),
    ]):
        data = []
        for variable in ["EP-flux-divergence", "zonal-mean-zonal-wind"]:
            if "600-200" in pressure_levels:
                filename = data_path / f"{reanalysis.lower()}_daily_{variable}_{pressure_levels}_{months_str}.nc"
            else:
                filename = data_path / f"{reanalysis.lower()}_daily_{variable}_{months_str}.nc"

            cube = iris.load_cube(filename, cs)

            cube = cube.aggregated_by("season_year", MEAN)[1:-1]
            data.append(cube)

        efp = efp_rolling_window(data[0], data[1], window_size)
        iplt.plot(efp, linestyle, label=f"{reanalysis}, {pressure_levels}")


def efp_rolling_window(ep_flux, u_zm, window_size):
    years = ep_flux.coord("season_year").points[:-window_size]

    efp = []
    for start_year in tqdm(years):
        cs = iris.Constraint(
            season_year=lambda x: start_year <= x <= start_year + window_size
        )

        ep_flux_sub = ep_flux.extract(cs).intersection(latitude=(25, 72))
        u_zm_sub = u_zm.extract(cs).intersection(latitude=(25, 72))

        corr = eddy_feedback_parameter(ep_flux_sub, u_zm_sub)

        if "pressure_level" in [c.name() for c in corr.coords()]:
            corr = corr.collapsed("pressure_level", MEAN)

        efp.append(corr.data)

    coord = DimCoord(points=years + (window_size + 1) // 2, long_name="start_year")
    efp = iris.cube.Cube(
        data=efp,
        long_name=f"eddy_feedback_parameter_{window_size}_year_window",
        dim_coords_and_dims=[(coord, 0)]
    )

    return efp


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
