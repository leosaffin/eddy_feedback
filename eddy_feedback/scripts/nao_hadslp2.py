import iris
from iris.cube import CubeList
from iris.util import equalise_attributes, unify_time_units
from iris.exceptions import ConcatenateError
from irise.grid import get_datetime

from constrain import north_atlantic_oscillation, regrid_to_coarsest

from eddy_feedback import datadir


def main():
    # The normal version ("") or the real time version ("r")
    variant = ""

    mslp = iris.load(datadir / f"constrain/hadslp2{variant}.pp")
    equalise_attributes(mslp)
    unify_time_units(mslp)

    years = []
    for cube in mslp:
        # The units attribute is not set but is clearly hPa
        cube.units = "hPa"

        # HADSLP2 loads in as a rotated grid with horizontal coordinates grid_longitude
        # and grid_latitude. The actual rotation of the grid is zero, so change the
        # coordinates for longitude/latitude and delete the coordinate system.
        # This makes things easier if we want to do any regridding or plotting on a map
        for axis, real_name in [("x", "longitude"), ("y", "latitude")]:
            coord = cube.coord(axis=axis, dim_coords=True)
            coord.coord_system = None
            coord.rename(real_name)

        # The cell_methods screws up merging the real time datasets
        cube.cell_methods = ()

        years += [c.year for c in get_datetime(cube)]

    try:
        mslp = mslp.concatenate_cube()
    except ConcatenateError:
        # The HADSLP2r version comes as three separate time periods, but cannot be
        # concatenated because one of the sets is in the middle of another. This can be
        # fixed by separating into a list of cubes by year and then concatenating
        cubes = CubeList()
        for year in set(years):
            for cube in mslp.extract(iris.Constraint(time=lambda x: x.point.year == year)):
                cubes.append(cube)
        mslp = cubes.concatenate_cube()

    for coord in ["latitude", "longitude"]:
        mslp.coord(coord).guess_bounds()

    iris.save(mslp, datadir / f"constrain/hadslp2{variant}.nc")
    mslp = iris.load_cube(datadir / f"constrain/hadslp2{variant}.nc")
    mslp = regrid_to_coarsest(mslp)
    nao = north_atlantic_oscillation.from_boxes(mslp)

    iris.save(nao, datadir / f"NAO_index_data/NAOI_monthly_all_HadSLP2{variant}_CanESM5-grid.nc")


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
