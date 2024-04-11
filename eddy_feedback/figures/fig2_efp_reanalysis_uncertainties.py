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
from iris.analysis import MEAN, STD_DEV
from iris.coords import DimCoord
import iris.plot as iplt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

from eddy_feedback import datadir, plotdir, bootstrapping
from eddy_feedback.nao_variance import season_mean
from eddy_feedback.figures import label_axes


data_path = datadir / "eddy_feedback/daily_mean"


def main():
    n_samples = 1000
    window_size = 23
    months = ["Dec", "Jan", "Feb"]

    fig, axes = plt.subplots(1, 2, sharey="all", figsize=(8, 5))

    panel_a(axes[0], n_samples, months)
    panel_b(axes[1], window_size, months)

    axes[0].set_xlabel("Sampling Period")
    axes[0].set_ylabel("Eddy-Feedback Parameter")
    axes[0].set_title("Sampling Uncertainty")

    axes[1].set_xlim(1912, 2011)
    axes[1].set_xlabel("Year")
    legend1 = axes[1].legend()
    axes[1].legend(
        handles=[
            Line2D([], [], color="C7", linestyle="-", label="EFP"),
            Line2D([], [], color="C7", linestyle="--", label=r"$\sigma^2$(NAO)"),
        ],
        loc="upper left", bbox_to_anchor=[0.0, 0.85],
    )
    axes[1].add_artist(legend1)
    axes[1].set_title("Multidecadal Variability")
    axes[1].axvline(2005, color="k")

    label_axes(axes.flatten())
    fig.suptitle("Eddy-Feedback Parameter - Reanalysis Uncertainties")
    fig.autofmt_xdate()

    plt.savefig(plotdir / f"fig2_efp_reanalysis_uncertainties.pdf")
    plt.show()


def panel_a(ax, n_samples, months):
    month_cs = iris.Constraint(month=months)
    year_blocks = [(1941, 2022), (1941, 1979), (1980, 2022)]

    # Store a list of each result labelled by the year range and pressure levels
    labels = []
    efp = []
    efp_full = []

    for plevs, suffix in [("500hPa", "NDJFM")]:
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
                length=end_year - start_year + 1,
                plevs=plevs,
            ))

    # Plot box-and-whisker plot for each time period
    positions = [1, 2, 3]
    ax.boxplot(efp, whis=(2.5, 97.5), medianprops=dict(color="C7"))
    ax.plot(positions, efp_full, "kx")
    ax.set_xticks(positions, labels)


def panel_b(ax, window_size, months):
    plt.axes(ax)
    cs = iris.Constraint(month=months)

    for n, (reanalysis, pressure_levels, months_str, linestyle) in enumerate([
        ("ERA5", "500hPa", "NDJFM", "-C7"),
        ("ERA20c", "500hPa", "DJF", "-C0"),
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
        print(efp.coord("centre_year").points)
        iplt.plot(efp, linestyle, label=reanalysis)

    ax2 = plt.twinx()
    nao_era5 = iris.load_cube(datadir / "NAO_index_data" / "NAOI_monthly_DJFM_ERA5.nc")
    nao_era5 = season_mean(nao_era5, months=months, seasons=["ndjfma", "mjjaso"])
    nao_era5 = nao_era5.rolling_window("season_year", STD_DEV, window_size) ** 2
    iplt.plot(nao_era5.coord("season_year"), nao_era5, "--C7", alpha=0.75)

    nao_era20c = iris.load_cube(datadir / "NAO_index_data" / "NAOI_monthly_all_ERA20C_CanESM5-grid.nc")
    nao_era20c = season_mean(nao_era20c, months=months, seasons=["ndjfma", "mjjaso"])
    nao_era20c = nao_era20c.rolling_window("season_year", STD_DEV, window_size) ** 2
    iplt.plot(nao_era20c.coord("season_year"), nao_era20c, "--C0", alpha=0.75)

    ax2.set_ylabel("NAO Variance (hPa)")


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

    coord = DimCoord(points=years + (window_size + 1) // 2, long_name="centre_year")
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
