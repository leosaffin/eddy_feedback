import matplotlib.pyplot as plt
import cmcrameri
import numpy as np
import iris
from iris.analysis import MEAN
import iris.plot as iplt
import cartopy.crs as ccrs

from eddy_feedback import datadir, plotdir, get_files_by_model
from eddy_feedback.local_eddy_feedback_north_atlantic_index import box
from eddy_feedback.figures import markers, label_axes


def main():
    projection = ccrs.EqualEarth()
    vmin = -1.15e-3
    vmax = 1.15e-3
    cmap = plt.get_cmap("cmc.vik")

    nrows, ncols = 5, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8), subplot_kw=dict(projection=projection))

    lefp_data = get_files_by_model("local_eddy_feedback_components", months="DJF")

    ensemble_means = []
    for n, model in enumerate(markers, start=3):
        print(model)

        lefp = []
        lefp_data_model = lefp_data.loc[lefp_data.model == model]

        for m, row in lefp_data_model.iterrows():
            lefp.append(iris.load_cube(
                row.filename, iris.Constraint(name="barotropic_energy_generation_rate")
            ))

        lefp_all = sum([cube.data for cube in lefp]) / len(lefp)
        lefp_all = lefp[0].copy(data=lefp_all)
        lefp_all = lefp_all.collapsed("time", MEAN)
        ensemble_means.append(lefp_all)

        plt.axes(axes[n // ncols, n % ncols])
        make_plot(lefp_all, model, na_box=box, vmin=vmin, vmax=vmax, cmap=cmap)

    all_models = sum([m.data for m in ensemble_means]) / len(ensemble_means)
    all_models = lefp_all.copy(data=all_models)
    plt.axes(axes[0, 2])
    make_plot(all_models, "All Models", na_box=box, vmin=vmin, vmax=vmax, cmap=cmap)

    plt.axes(axes[0, 0])
    x = all_models.coord("longitude").points
    y = all_models.coord("latitude").points

    era5 = iris.load_cube(
        datadir / "local_eddy_feedback_parameter_data/daily" / "G_mean_FY02_lat_lon_DJF_ERA5.nc",
    )
    era5 = era5.intersection(longitude=(x.min(), x.max()), latitude=(y.min(), y.max()))
    print((x.min(), x.max()), (y.min(), y.max()))
    era5 = era5.extract(iris.Constraint(pressure_level=250))
    era5 = era5.collapsed("time", MEAN)

    im = make_plot(era5, "ERA5", na_box=box, vmin=vmin, vmax=vmax, cmap=cmap)

    label_axes(np.concatenate(([axes[0, 0]], axes.flatten()[2:])))
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.05, 0.1, 0.9, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Barotropic energy generation rate (m$^2$ s$^{-3}$)")

    axes[0, 1].set_visible(False)

    plt.savefig(plotdir / "figA1_lefp_NA_cmip6.pdf")
    plt.show()


def make_plot(lefp, model, na_box=(-60, -25, 30, 45), **kwargs):
    print(lefp.data.min(), lefp.data.max())
    im = iplt.pcolormesh(lefp, **kwargs)
    ax = plt.gca()
    ax.coastlines()

    transform = ccrs.PlateCarree()
    ax.plot([na_box[0], na_box[1]], [na_box[2], na_box[2]], "-k", transform=transform)
    ax.plot([na_box[1], na_box[1]], [na_box[2], na_box[3]], "-k", transform=transform)
    ax.plot([na_box[1], na_box[0]], [na_box[3], na_box[3]], "-k", transform=transform)
    ax.plot([na_box[0], na_box[0]], [na_box[3], na_box[2]], "-k", transform=transform)

    ax.set_title(f"{model}")

    return im


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
