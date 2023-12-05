"""
Calculate a North Atlantic index for the local eddy feedback by averaging over a box
for the seasonal mean
"""


import iris
from iris.analysis import MEAN

from constrain.north_atlantic_oscillation import box_average

from eddy_feedback import get_files_by_model, datadir
from eddy_feedback.nao_variance import season_mean

output_filename = str(
    datadir / "local_eddy_feedback_parameter_data/north_atlantic_index/"
              "G-mean_north-atlantic_DJF_{model}_historical_{variant}.nc"
)
output_filename_era5 = output_filename.format(
    model="ERA5", variant=""
).replace("_historical_", "")

box = [-60, -25, 30, 45]


def main():
    months = ["Dec", "Jan", "Feb"]
    seasons = ["ndjfma", "mjjaso"]

    lefp = get_files_by_model("local_eddy_feedback_monthly", months="DJFM")

    for n, row in lefp.iterrows():
        model = row["model"]
        variant = row["variant"]

        print(model, variant)

        lefp = iris.load_cube(row.filename)
        lefp = season_mean(lefp, months=months, seasons=seasons)
        mean = box_average(lefp, box)

        iris.save(mean, output_filename.format(model=model, variant=variant))

    # Repeat for ERA5
    # Slightly different set of steps because the data is stored daily rather than
    # monthly
    lefp = iris.load_cube(
        datadir / "local_eddy_feedback_parameter_data/daily/G_mean_MC89_lat_lon_DJF_ERA5.nc",
        iris.Constraint(month=months)
    )
    lefp = lefp.aggregated_by("season_year", MEAN)
    mean = box_average(lefp, box)
    iris.save(mean, output_filename_era5)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
