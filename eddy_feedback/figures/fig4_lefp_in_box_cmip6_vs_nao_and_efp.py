from string import ascii_lowercase

import iris
from iris.analysis import MEAN
import numpy as np
import matplotlib.pyplot as plt

from eddy_feedback import plotdir, bootstrapping, local_eddy_feedback_north_atlantic_index
from eddy_feedback.figures.fig3_efp_vs_nao_cmip6 import markers, get_data


def main():
    months = ["Dec", "Jan", "Feb"]
    start_year, end_year = 1941, 2022

    months_str = "".join([m[0] for m in months])
    data = get_data(months_str, "1850-2014")

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey="all")
    models = sorted(set(data.model), key=lambda x: x.lower())
    for model in models:
        data_model = data.loc[data.model == model]
        print(model)

        # Keep values of G to calculate ensemble mean
        lefp_all = []

        # Ensemble members
        for n, row in data_model.iterrows():
            # Load North-Atlantic box averaged G
            fname = local_eddy_feedback_north_atlantic_index.output_filename.format(
                model=row.model, variant=row.variant
            )
            try:
                lefp = iris.load_cube(fname)
                lefp = lefp.collapsed("season_year", MEAN).data
                lefp_all.append(lefp)
                axes[0].plot(row.efp, lefp, markers[model], alpha=0.5)
                axes[1].plot(row.nao_variance, lefp, markers[model], alpha=0.5)
            except OSError:
                print(row.model, row.variant, "has no corresponding data")

        # Ensemble mean
        if len(lefp_all) > 0:
            lefp_mean = np.mean(lefp_all)
            axes[0].plot(
                np.mean(data_model.efp), lefp_mean, markers[model], mec="k", label=model
            )
            axes[1].plot(
                np.mean(data_model.nao_variance), lefp_mean, markers[model], mec="k"
            )

    # Add ERA5
    nao_variance_era5_samples = bootstrapping.bootstrap_nao(
        start_year=start_year, end_year=end_year, n_samples=1000, months=months, months_str=months_str,
    )
    efp_era5_samples = bootstrapping.bootstrap_eddy_feedback_parameter(
        start_year=start_year, end_year=end_year, n_samples=1000, plevs="500hPa"
    )
    lefp_era5_samples = bootstrapping.bootstrap_local_eddy_feedback_parameter(
        start_year=start_year, end_year=end_year, n_samples=1000, plevs="250hPa",
    )

    axes[0].plot(efp_era5_samples, lefp_era5_samples, ".k", alpha=0.1)
    axes[1].plot(nao_variance_era5_samples, lefp_era5_samples, ".k", alpha=0.1)

    axes[0].set_ylabel("Barotropic Energy Generation Rate (m$^2$ s$^{-3}$)")
    axes[0].set_xlabel("Eddy-Feedback Parameter")
    axes[1].set_xlabel("NAO Variance (hPa)")
    fig.subplots_adjust(bottom=0.25)
    fig.legend(loc="center", ncol=4, bbox_to_anchor=(0.5, 0.1))

    for n, ax in enumerate(axes.flatten()):
        ax.text(0.01, 1.02, f"({ascii_lowercase[n]})", transform=ax.transAxes)

    plt.savefig(plotdir / "fig4_lefp-in-box-cmip6_vs_efp_and_nao")
    plt.show()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
