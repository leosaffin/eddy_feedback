import iris
from iris.analysis import MEAN, STD_DEV
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

from eddy_feedback import datadir, plotdir, get_reanalysis_diagnostic, bootstrapping, local_eddy_feedback_north_atlantic_index
from eddy_feedback.nao_variance import season_mean
from eddy_feedback.figures import markers, label_axes
from eddy_feedback.figures.fig3_efp_vs_nao_cmip6 import weighted_average_regression, print_as_latex


def main():
    months = ["Dec", "Jan", "Feb"]
    start_year, end_year = 1941, 2022
    n_samples = 1000
    length = 164

    months_str = "".join([m[0] for m in months])
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharey="row")

    # Add ERA5
    nao_variance_era5_samples = bootstrapping.bootstrap_nao(
        start_year=start_year, end_year=end_year, n_samples=n_samples, length=length, months=months, months_str=months_str,
    )
    efp_era5_samples = bootstrapping.bootstrap_eddy_feedback_parameter(
        start_year=start_year, end_year=end_year, n_samples=n_samples, length=length, plevs="500hPa"
    )
    lefp_era5_samples = bootstrapping.bootstrap_local_eddy_feedback_parameter(
        start_year=start_year, end_year=end_year, n_samples=n_samples, length=length,  plevs="250hPa",
    )
    axes[2, 1].plot(lefp_era5_samples, efp_era5_samples, ".C7", alpha=0.1)
    axes[0, 1].plot(lefp_era5_samples, nao_variance_era5_samples, ".C7", alpha=0.1)

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
    lefp_full = lefp_era5.collapsed("season_year", MEAN).data
    print(lefp_full)

    lefp_jra55 = iris.load_cube(local_eddy_feedback_north_atlantic_index.output_filename_era5.replace("ERA5", "JRA55"))
    lefp_jra_full = lefp_jra55.collapsed("season_year", MEAN).data
    print(lefp_jra_full)

    axes[2, 1].plot(lefp_full, efp_full, "oC7", mec="k")
    axes[0, 1].plot(lefp_full, nao_full, "oC7", mec="k")
    #axes[2].plot(lefp_full, nao_multidecadal, ".k", mec="grey")

    kde_ax_x = axes[0, 1].twinx()
    kde_ax_x.yaxis.set_visible(False)
    kde_ax_x_2 = axes[2, 1].twinx()
    kde_ax_x_2.yaxis.set_visible(False)
    kde_ax_y = axes[2, 1].twiny()
    kde_ax_y.xaxis.set_visible(False)
    kde_ax_y_2 = axes[0, 1].twiny()
    kde_ax_y_2.xaxis.set_visible(False)

    color = "C7"
    seaborn.kdeplot(x=lefp_era5_samples, ax=kde_ax_x, color=color, fill=True, alpha=0.5)
    seaborn.kdeplot(x=lefp_era5_samples, ax=kde_ax_x_2, color=color, fill=True, alpha=0.5)
    seaborn.kdeplot(y=efp_era5_samples, ax=kde_ax_y, color=color, fill=True, alpha=0.5)
    seaborn.kdeplot(y=nao_variance_era5_samples, ax=kde_ax_y_2, color=color, fill=True, alpha=0.5)

    kde_ax_x.set_ylim(0, 200000)
    kde_ax_x_2.set_ylim(0, 200000)
    kde_ax_y.set_xlim(50, 0)
    kde_ax_y_2.set_xlim(1, 0)

    # Add CMIP6 models
    data = pd.read_csv(datadir / "CMIP6_diagnostics_by_model.csv")
    models = sorted(set(data.model), key=lambda x: x.lower())
    for model in models:
        data_model = data.loc[data.model == model]
        # Ensemble mean
        lefp_mean = np.mean(data_model.G_na)

        for n, diag in enumerate(["nao_variance", "nao_variance_multidecadal", "efp"]):
            if n == 2:
                label = model
            else:
                label = None
            # Ensemble mean
            axes[n, 0].plot(
                lefp_mean, np.mean(data_model[diag]), markers[model], mec="k", zorder=10, label=label,
            )

            # Ensemble members
            axes[n, 0].plot(
                data_model.G_na, data_model[diag], markers[model], alpha=0.5
            )

    # Linear regressions
    xp = np.arange(-0.001, 0.0, 0.00001)
    lefp_mean = [np.mean(data.loc[data.model == model].G_na) for model in markers]
    for n, diag in enumerate(["nao_variance", "nao_variance_multidecadal", "efp"]):
        print("\n" + diag)
        diag_mean = [np.mean(data.loc[data.model == model][diag]) for model in markers]
        result_mean = linregress(lefp_mean, diag_mean)
        result_all = linregress(data.G_na, data[diag])
        result_weighted = weighted_average_regression(data, markers.keys(), "G_na", diag)

        print_as_latex(result_mean, "Mean")
        print_as_latex(result_all, "All")
        print_as_latex(result_weighted, "Weighted")

        for result, linestyle in [
            (result_mean, "-k"),
            (result_all, "--k"),
            (result_weighted, ":k"),
        ]:
            axes[n, 0].plot(xp, result.slope * xp + result.intercept, linestyle, alpha=0.75)
            if "multidecadal" not in diag:
                axes[n, 1].plot(xp, result.slope * xp + result.intercept, linestyle, alpha=0.75)

    print("\nERA5")
    result = linregress(lefp_era5_samples, nao_variance_era5_samples)
    print_as_latex(result, "ERA5")
    axes[0, 1].plot(xp, result.slope * xp + result.intercept, "-C7", alpha=0.75)
    result = linregress(lefp_era5_samples, efp_era5_samples)
    print_as_latex(result, "ERA5")
    axes[2, 1].plot(xp, result.slope * xp + result.intercept, "-C7", alpha=0.75)

    axes[0, 0].set_title("CMIP6")
    axes[0, 1].set_title("ERA5")
    axes[2, 0].set_xlabel(r"$G_\mathrm{NA}$ (m$^2$ s$^{-3}$)")
    axes[0, 1].set_xlabel(r"$G_\mathrm{NA}$ (m$^2$ s$^{-3}$)")
    axes[2, 1].set_xlabel(r"$G_\mathrm{NA}$ (m$^2$ s$^{-3}$)")
    axes[2, 0].set_ylabel("EFP")
    axes[0, 0].set_ylabel("NAO Variance (hPa)\nTotal")
    axes[1, 0].set_ylabel("NAO Variance (hPa)\n20-Year Filter")
    axes[1, 1].set_visible(False)

    custom_lines = [
        Line2D([0], [0], linestyle=linestyle, color="k") for linestyle in ["-", "--", ":"]]
    labels = ["Mean", "All", "Weighted"]

    l = axes[0, 0].legend(custom_lines, labels, framealpha=0.25)
    l.set_zorder(11)
    custom_lines = [
        Line2D([], [], marker=markers[model][0], color=markers[model][1:], linestyle="", mec="k")
        for model in markers]
    fig.legend(custom_lines, markers.keys(), ncol=2, loc="center", bbox_to_anchor=(0.7, 0.5))

    for ax in axes.flatten():
        ax.set_xlim(-0.0009, -0.0003)
    axes[0, 0].xaxis.set_ticklabels([])
    axes[1, 0].xaxis.set_ticklabels([])

    label_axes(np.concatenate((axes.flatten()[:3], axes.flatten()[4:]), axis=None))

    plt.savefig(plotdir / "fig4_lefp-in-box-cmip6_vs_efp_and_nao.pdf")
    plt.show()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
