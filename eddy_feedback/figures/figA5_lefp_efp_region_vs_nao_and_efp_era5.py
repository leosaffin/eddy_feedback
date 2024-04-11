import iris
from iris.analysis import MEAN, STD_DEV
import matplotlib.pyplot as plt
import seaborn

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

from eddy_feedback import (
    datadir, plotdir, get_reanalysis_diagnostic, bootstrapping,
    local_eddy_feedback_north_atlantic_index, local_eddy_feedback_efp_region_index
)
from eddy_feedback.nao_variance import season_mean
from eddy_feedback.figures import label_axes
from eddy_feedback.figures.fig4_lefp_in_box_cmip6_vs_nao_and_efp import (
    months, start_year, end_year, n_samples, length
)


def main():
    months_str = "".join([m[0] for m in months])
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")

    # Add ERA5
    nao_variance_era5_samples = bootstrapping.bootstrap_nao(
        start_year=start_year, end_year=end_year, n_samples=n_samples, length=length, months=months, months_str=months_str,
    )
    efp_era5_samples = bootstrapping.bootstrap_eddy_feedback_parameter(
        start_year=start_year, end_year=end_year, n_samples=n_samples, length=length, plevs="500hPa"
    )

    lefp_era5_samples = bootstrapping.bootstrap_local_eddy_feedback_parameter(
        start_year=start_year, end_year=end_year, n_samples=n_samples, length=length, plevs="250hPa",
    )
    lefp_era5_samples_efp = bootstrapping.bootstrap_local_eddy_feedback_parameter_efp_region(
        start_year=start_year, end_year=end_year, n_samples=n_samples, length=length,  plevs="250hPa",
    )

    # Add full values
    data_path = datadir / "eddy_feedback/daily_mean/"
    month_cs = iris.Constraint(month=months)
    ep_flux = iris.load_cube(data_path / f"era5_daily_EP-flux-divergence_NDJFM.nc", month_cs)
    ep_flux = ep_flux.aggregated_by("season_year", MEAN)[1:-1]
    u_zm = iris.load_cube(data_path / f"era5_daily_zonal-mean-zonal-wind_NDJFM.nc", month_cs)
    u_zm = u_zm.aggregated_by("season_year", MEAN)[1:-1]
    nao = get_reanalysis_diagnostic("north_atlantic_oscillation", months="DJFM")
    nao = season_mean(nao, months=months, seasons=["ndjfma", "mjjaso"])
    efp_full = eddy_feedback_parameter(ep_flux, u_zm).data
    nao_full = nao.collapsed("season_year", STD_DEV).data ** 2
    nao_multidecadal = nao.rolling_window("season_year", MEAN, 20)
    nao_multidecadal = nao_multidecadal.collapsed("season_year", STD_DEV).data ** 2
    lefp_era5 = iris.load_cube(local_eddy_feedback_north_atlantic_index.output_filename_era5, iris.Constraint(pressure_level=250))
    lefp_efp_era5 = iris.load_cube(local_eddy_feedback_efp_region_index.output_filename_era5, iris.Constraint(pressure_level=250))
    lefp_full = lefp_era5.collapsed("season_year", MEAN).data
    lefp_efp_full = lefp_efp_era5.collapsed("season_year", MEAN).data
    print(lefp_full)
    print(lefp_efp_full)

    lefp_jra55 = iris.load_cube(local_eddy_feedback_efp_region_index.output_filename_era5.replace("ERA5", "JRA55"))
    lefp_jra_full = lefp_jra55.collapsed("season_year", MEAN).data
    print(lefp_jra_full)

    axes[0, 0].plot(lefp_full, nao_full, "oC7", mec="k", zorder=1000)
    axes[1, 0].plot(lefp_full, efp_full, "oC7", mec="k", zorder=1000)
    axes[0, 1].plot(lefp_efp_full, nao_full, "oC7", mec="k", zorder=1000)
    axes[1, 1].plot(lefp_efp_full, efp_full, "oC7", mec="k", zorder=1000)

    kde_ax_x = [[axes[0, 0].twinx(), axes[0, 1].twinx()], [axes[1, 0].twinx(), axes[1, 1].twinx()]]
    kde_ax_y = [[axes[0, 0].twiny(), axes[0, 1].twiny()], [axes[1, 0].twiny(), axes[1, 1].twiny()]]
    for axes_set in [kde_ax_x, kde_ax_y]:
        for axes_ in axes_set:
            for ax in axes_:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

    color = "C7"
    for n, lefp_samples in enumerate([lefp_era5_samples, lefp_era5_samples_efp]):
        axes[0, n].plot(lefp_samples, nao_variance_era5_samples, ".C7", alpha=0.1)
        axes[1, n].plot(lefp_samples, efp_era5_samples, ".C7", alpha=0.1)
        seaborn.kdeplot(x=lefp_samples, ax=kde_ax_x[0][n], color=color, fill=True, alpha=0.5)
        seaborn.kdeplot(x=lefp_samples, ax=kde_ax_x[1][n], color=color, fill=True, alpha=0.5)
    seaborn.kdeplot(y=nao_variance_era5_samples, ax=kde_ax_y[0][1], color=color, fill=True, alpha=0.5)
    seaborn.kdeplot(y=efp_era5_samples, ax=kde_ax_y[1][1], color=color, fill=True, alpha=0.5)

    axes[1, 0].set_xlabel(r"$G_\mathrm{NA}$ (m$^2$ s$^{-3}$)")
    axes[1, 1].set_xlabel(r"$G_\mathrm{EFP}$ (m$^2$ s$^{-3}$)")
    axes[1, 0].set_ylabel("EFP")
    axes[0, 0].set_ylabel("NAO Variance (hPa)\nTotal")

    label_axes(axes)

    # Limits for distributions set so they don't overlap scatter points but still stay
    # on the same (twin) axes
    kde_ax_x[0][0].set_ylim(0, 200000)
    kde_ax_x[1][0].set_ylim(0, 200000)
    kde_ax_x[0][1].set_ylim(0, 1500000)
    kde_ax_x[1][1].set_ylim(0, 1500000)
    kde_ax_y[1][1].set_xlim(50, 0)
    kde_ax_y[0][1].set_xlim(1, 0)
    axes[0, 0].set_xlim(-0.0009, -0.0003)

    # Scale axes so they have the same relative xlimits to the full ERA5 value for both
    # plots of G
    scaling_min = lefp_full / -0.0009
    scaling_max = lefp_full / -0.0003
    xmin = lefp_efp_full / scaling_min
    xmax = lefp_efp_full / scaling_max
    axes[0, 1].set_xlim(xmin, xmax)

    # Match limits to Fig. 4 printed out
    axes[0, 0].set_ylim(4.830177046070203, 28.357392865129846)
    axes[1, 0].set_ylim(0.08139575644115388, 0.39074220207423077)

    plt.savefig(plotdir / "figA5_lefp-in-box_vs_lefp-efp-region.png")
    plt.show()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
