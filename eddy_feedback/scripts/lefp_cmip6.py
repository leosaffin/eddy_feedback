"""
Calculation of LEFP for CMIP6 data ESGF downloaded to Leeds servers
"""


import pathlib

import iris
import iris.util
from iris.coord_categorisation import add_month, add_season_year
from parse import parse
from tqdm import tqdm

from constrain import eddy_feedback_parameter


filename_pattern = "{variable}_day_{model}_historical_{runid}_{plev}hPa_{months}_rg_NATL_mergetime.nc"


def main():
    months = ['Dec', 'Jan', 'Feb']
    plev = 250
    wind_months = "NDJFMA"

    # Load in directories
    datadir = pathlib.Path(
        "/nfs/annie/earcmc/CMIP6_LEs/"
        "eddy_feedback_Mak_and_Cai/CMIP6_hist/daily_ua_va/saved_nc_files/ESGF_JASMIN/"
    )
    savedir = pathlib.Path(
        "/nfs/annie/earlsa/eddy_feedback_parameter/local_eddy_feedback/CMIP6/"
    )

    models = [path.name for path in datadir.glob("*")]
    # Loop through simulations, calculating the eddy feedback
    for model in models:
        print(model)

        files = (datadir / model).glob(filename_pattern.format(
            variable="ua", model=model, runid="*", plev=plev, months=wind_months,
        ))
        runids = [parse(filename_pattern, f.name).named["runid"] for f in files]
        for runid in runids:
            print(runid)

            output_file = savedir / f"G_mean_lat_lon_DJF_{model}_historical_{runid}.nc"
            if not output_file.exists():
                print(model, runid)

                # Load in ua and va data for the current record
                ua = iris.load_cube(datadir / model / filename_pattern.format(
                    variable="ua", model=model, runid=runid, plev=plev, months=wind_months,
                ))
                va = iris.load_cube(datadir / model / filename_pattern.format(
                    variable="va", model=model, runid=runid, plev=plev, months=wind_months,
                ))

                # Remove dimensions of length 1
                ua = iris.util.squeeze(ua)
                va = iris.util.squeeze(va)

                for cube in [ua, va]:
                    add_month(cube, "time")
                    add_season_year(cube, "time", seasons=["ndjfma", "mjjaso"])

                lefp = calc_local_eddy_feedback(ua, va, months)
                iris.save(lefp, output_file)


def calc_local_eddy_feedback(ua, va, months):
    # Calculate by year to not use too much memory
    # Ignore first and last year as they are not full seasons
    lefp_all_years = iris.cube.CubeList()
    season_years = list(set(ua.coord("season_year").points))[1:-1]
    for year in tqdm(season_years):
        cs = iris.Constraint(season_year=year)
        ua_s = ua.extract(cs)
        va_s = va.extract(cs)

        # Calculate E-vectors
        e_lon, e_lat = eddy_feedback_parameter.Evectors(
            ua_s, va_s, months=months, window=61, f1=2, f2=6
        )

        # Calculate background deformation flow, D
        d_lon, d_lat = eddy_feedback_parameter.background_deformation_flow(
            ua_s, va_s, months=months, window=61, f=10
        )

        # Calculate barotropic energy generation rate, G=E.D, and
        # correct metadata
        G = e_lon * d_lon + e_lat * d_lat
        G.rename('barotropic_energy_generation_rate')
        G.units = 'm2 s-3'

        lefp_all_years.append(G)

    return lefp_all_years.concatenate_cube()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
