"""
Calculate a North Atlantic index for the local eddy feedback by averaging over a box
for the seasonal mean
"""

import iris

from constrain.north_atlantic_oscillation import box_average

from eddy_feedback import get_files_by_model, datadir

output_filename = str(
    datadir / "local_eddy_feedback_parameter_data/efp_region_index/"
              "G-mean_efp_region_DJF_{model}_historical_{variant}.nc"
)
output_filename_era5 = output_filename.format(
    model="ERA5", variant=""
).replace("_historical_", "")
output_filename_jra55 = output_filename.format(
    model="JRA55", variant=""
).replace("_historical_", "")

box = [-180, 180, 25, 72]


def main():
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
