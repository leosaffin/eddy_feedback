from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
import cmcrameri
import iris
from iris.analysis import MEAN
import iris.plot as iplt
import cartopy.crs as ccrs

from eddy_feedback import datadir, plotdir
from eddy_feedback.nao_variance import season_mean
from eddy_feedback.local_eddy_feedback_north_atlantic_index import box
from eddy_feedback.figures import markers


def main():
    fname = str(
        datadir / "local_eddy_feedback_parameter_data/Monthly_local_EFP_IPSL/"
                  "G_lat_lon_DJFM_{model}_historical_r*_monthly.nc"
    )

    projection = ccrs.EqualEarth()
    vmin = -1.5e-3
    vmax = 1.5e-3
    cmap = plt.get_cmap("cmc.vik")

    nrows, ncols = 5, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8), subplot_kw=dict(projection=projection))

    all_models = iris.cube.CubeList()
    n_models = 0
    for n, model in enumerate(markers, start=2):
        print(model)
        lefp = iris.load(fname.format(model=model))
        n_members = len(lefp)
        lefp_all = sum([cube.data for cube in lefp]) / n_members
        lefp_all = lefp[0].copy(data=lefp_all)

        lefp_all = season_mean(lefp_all, months=["Dec", "Jan", "Feb"], seasons=["ndjfma", "mjjaso"])
        lefp_all = lefp_all.collapsed("time", MEAN)

        all_models.append(lefp_all * n_members)
        n_models += n_members

        plt.axes(axes[n // ncols, n % ncols])
        make_plot(lefp_all, model, n, vmin=vmin, vmax=vmax, cmap=cmap)

    all_models = sum([m.data for m in all_models]) / n_models
    all_models = lefp_all.copy(data=all_models)
    plt.axes(axes[0, 1])
    make_plot(all_models, "All Models", 1, vmin=vmin, vmax=vmax, cmap=cmap)

    plt.axes(axes[0, 0])
    x = all_models.coord("longitude").points
    y = all_models.coord("latitude").points

    era5 = iris.load_cube(
        datadir / "local_eddy_feedback_parameter_data/daily" / "G_mean_MC89_lat_lon_DJF_ERA5.nc",
        iris.Constraint(month=["Dec", "Jan", "Feb"])
    )
    era5 = era5.intersection(longitude=(x.min(), x.max()), latitude=(y.min(), y.max()))
    print((x.min(), x.max()), (y.min(), y.max()))
    era5 = era5.extract(iris.Constraint(pressure_level=250))
    era5 = era5.aggregated_by("season_year", MEAN)
    era5 = era5.collapsed("time", MEAN)

    im = make_plot(era5, "ERA5", 0, vmin=vmin, vmax=vmax, cmap=cmap)

    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.05, 0.1, 0.9, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Barotropic energy generation rate (m$^2$ s$^{-3}$)")

    plt.savefig(plotdir / "figA1_lefp_NA_cmip6.png")
    plt.show()


def make_plot(lefp, model, n, na_box=(-60, -25, 30, 45), **kwargs):
    im = iplt.pcolormesh(lefp, **kwargs)
    ax = plt.gca()
    ax.coastlines()

    transform = ccrs.PlateCarree()
    ax.plot([na_box[0], na_box[1]], [na_box[2], na_box[2]], "-k", transform=transform)
    ax.plot([na_box[1], na_box[1]], [na_box[2], na_box[3]], "-k", transform=transform)
    ax.plot([na_box[1], na_box[0]], [na_box[3], na_box[3]], "-k", transform=transform)
    ax.plot([na_box[0], na_box[0]], [na_box[3], na_box[2]], "-k", transform=transform)

    ax.text(-0.1, 1.1, f"({ascii_lowercase[n]})", transform=ax.transAxes)
    ax.set_title(f"{model}")

    return im


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
