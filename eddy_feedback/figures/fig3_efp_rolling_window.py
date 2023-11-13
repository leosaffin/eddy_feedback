from tqdm import tqdm
import iris
from iris.analysis import MEAN
from iris.coords import DimCoord
import iris.plot as iplt
import matplotlib.pyplot as plt

from constrain import eddy_feedback_parameter

from eddy_feedback import datadir, plotdir


def main():
    data_path = datadir / "eddy_feedback/daily_mean/"
    window_size = 23
    months = ["Dec", "Jan", "Feb"]

    plt.figure(figsize=(8, 5))
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

    plt.xlabel("Year")
    plt.ylabel(r"$r^2$")
    plt.legend()
    plt.title("Eddy-Feedback Parameter for {}-Year Rolling Window".format(window_size))
    plt.axvline(2005, color="k")

    plt.savefig(plotdir / "fig3_eddy-feedback-parameter_rolling-window.png")
    plt.show()


def efp_rolling_window(ep_flux, u_zm, window_size):
    years = ep_flux.coord("season_year").points[:-window_size]

    efp = []
    for start_year in tqdm(years):
        cs = iris.Constraint(
            season_year=lambda x: start_year <= x <= start_year + window_size
        )

        ep_flux_sub = ep_flux.extract(cs).intersection(latitude=(25, 72))
        u_zm_sub = u_zm.extract(cs).intersection(latitude=(25, 72))

        corr = eddy_feedback_parameter.eddy_feedback_parameter(
            ep_flux_sub, u_zm_sub
        )

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
