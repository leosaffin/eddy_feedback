"""
Calculate a North Atlantic index for the local eddy feedback by averaging over a box
for the seasonal mean
"""

import iris

from constrain.north_atlantic_oscillation import box_average

from eddy_feedback import get_files_by_model, datadir

output_filename = str(
    datadir / "local_eddy_feedback_parameter_data/north_atlantic_index/"
              "G-mean_north-atlantic_DJF_{model}_historical_{variant}.nc"
)
output_filename_era5 = output_filename.format(
    model="ERA5", variant=""
).replace("_historical_", "")
output_filename_jra55 = output_filename.format(
    model="JRA55", variant=""
).replace("_historical_", "")

box = [-60, -25, 30, 45]


def main():
    # CMIP6 from DJF-mean data
    lefp_data = get_files_by_model("local_eddy_feedback_components", months="DJF")

    for n, row in lefp_data.iterrows():
        model = row["model"]
        variant = row["variant"]

        print(model, variant)
        fname = output_filename.format(model=model, variant=variant)

        lefp = iris.load_cube(row.filename, "barotropic_energy_generation_rate")

        if lefp.coord("season_year").points[0] < 1900:
            mean = box_average(lefp, box)
            iris.save(mean, fname)
        else:
            print("Not full historical period")

    # ERA5
    path = datadir / "local_eddy_feedback_parameter_data/daily/"
    lefp = iris.load_cube(path / "G_mean_FY02_lat_lon_DJF_ERA5.nc")
    mean = box_average(lefp, box)
    iris.save(mean, output_filename_era5)

    # JRA55
    lefp = iris.load_cube(path / "G_mean_FY02_lat_lon_DJF_JRA55.nc")
    mean = box_average(lefp, box)
    iris.save(mean, output_filename_jra55)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
