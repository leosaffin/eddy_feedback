import iris

from constrain import north_atlantic_oscillation

from eddy_feedback import datadir


def main():
    mslp = iris.load_cube(datadir / f"constrain/ERA20C/prmsl.mon.mean.nc")
    for coord in ["longitude", "latitude"]:
        mslp.coord(coord).guess_bounds()
    nao = north_atlantic_oscillation.from_boxes(mslp)
    iris.save(nao, datadir / f"NAO_index_data/NAOI_monthly_all_ERA20C.nc")


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
